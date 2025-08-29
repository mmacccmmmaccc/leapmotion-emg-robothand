import leap
import numpy as np
import cv2
import serial

_TRACKING_MODES = {
    leap.TrackingMode.Desktop: "Desktop",
    leap.TrackingMode.HMD: "HMD",
    leap.TrackingMode.ScreenTop: "ScreenTop", 
}


class Canvas:
    def __init__(self, ser):
        self.ser = ser
        self.last_command = None
        self.name = "Python Gemini Visualiser"
        self.screen_size = [500, 700]
        self.hands_colour = (255, 255, 255)
        self.font_colour = (0, 255, 44)
        self.hands_format = "Skeleton"
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        self.tracking_mode = None

    def set_tracking_mode(self, tracking_mode):
        self.tracking_mode = tracking_mode

    def toggle_hands_format(self):
        self.hands_format = "Dots" if self.hands_format == "Skeleton" else "Skeleton"
        print(f"Set hands format to {self.hands_format}")

    def get_joint_position(self, bone):
        if bone:
            return int(bone.x + (self.screen_size[1] / 2)), int(bone.z + (self.screen_size[0] / 2))
            # return int(bone.x + (self.screen_size[1] / 2)), int(-bone.y + (self.screen_size[0] / 2))
        else:
            return None

    def draw_palm(self, hand):
        palm_pos = hand.palm.position
        palm_2d = self.get_joint_position(palm_pos)  # convert to 2D canvas coordinates
        if palm_2d:
            cv2.circle(self.output_image, palm_2d, 6, (0, 0, 255), -1) # palm red dot
            text = f"Palm: x={palm_pos.x:.1f}, y={palm_pos.y:.1f}, z={palm_pos.z:.1f}" # palm position text
            text_pos = (palm_2d[0] + 40, palm_2d[1])
            cv2.putText(
                self.output_image,
                text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return palm_2d
    
    def get_finger_openness(self, finger):
        """
        Estimate finger openness (0=closed, 1=fully extended)
        by calculating angle between proximal and intermediate bones.
        """
        def vector_from_bones(start, end):
            return np.array([end.x - start.x, end.y - start.y, end.z - start.z])

        proximal_bone = finger.bones[1]      # proximal phalanx
        intermediate_bone = finger.bones[2]  # intermediate phalanx

        v1 = vector_from_bones(proximal_bone.prev_joint, proximal_bone.next_joint)
        v2 = vector_from_bones(intermediate_bone.prev_joint, intermediate_bone.next_joint)

        v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)

        cosine_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(cosine_angle)  # radians

        openness = 1 - (angle / np.pi)  # map angle to [0,1] openness
        return openness
    
    def get_finger_openness_tip_to_palm(self, finger, palm_position, min_dist=25.0, max_dist=70.0):
        # Use index 3 for distal bone (tip)
        tip_pos = finger.bones[3].next_joint

        dist = ((tip_pos.x - palm_position.x) ** 2 +
                (tip_pos.y - palm_position.y) ** 2 +
                (tip_pos.z - palm_position.z) ** 2) ** 0.5

        openness = (dist - min_dist) / (max_dist - min_dist)
        openness = max(0.0, min(1.0, openness))

        return openness


    def get_fingers_openness(self, hand):
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        openness_list = []

        palm_position = hand.palm.position

        # Calibrate min/max distances per finger if you want (same here for simplicity)
        min_dist = 25.0  # mm (adjust if needed)
        max_dist = 70.0  # mm (adjust if needed)

        for i, finger in enumerate(hand.digits):
            openness = self.get_finger_openness_tip_to_palm(finger, palm_position, min_dist, max_dist)
            openness_list.append((finger_names[i], openness))
        return openness_list


    # def get_fingers_openness(self, hand):
    #     finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    #     openness_list = []
    #     for i, finger in enumerate(hand.digits):
    #         openness = self.get_finger_openness(finger)
    #         openness_list.append((finger_names[i], openness))
    #     return openness_list

    def send_finger_openness_uart(self, hand):  
        finger_openness = self.get_fingers_openness(hand)  # list of (finger_name, openness) [(0, 0), (1,0.6)]
        # 0-100% openness values
        # values = [str(int(openness*100)) for _, openness in finger_openness] 

        # Round to nearest 10% for UART
        values_list = []

        for _, openness in finger_openness:
            values = round(int(openness * 100) / 10) * 10
            angle =  int(((values/100) * 120) + 60)
            values_list.append(str(angle))

        send_str = ','.join(values_list) + '\n'
        
        # To avoid spamming same data repeatedly, send only if changed
        if send_str != self.last_command:
            print(f"Sending UART: {send_str.strip()}")
            if self.ser is not None:
                try:
                    self.ser.write(send_str.encode('utf-8'))
                    self.ser.flush()
                except Exception as e:
                    print(f"[UART ERROR] {e}")
            else:
                print("[UART WARNING] Serial port not connected, not sending.")
            self.last_command = send_str


    def draw_finger_info(self, hand, pos=(10, 20)):
        hand_type = "Right" if hand.type == leap.HandType.Right else "Left"
        grab_strength = hand.grab_strength  # 0 (open) to 1 (closed)

        finger_openness = self.get_fingers_openness(hand)
        count_open = sum(1 for _, op in finger_openness if op > 0.5)

        if count_open == 5 and grab_strength <= 0.8:
            hand_status = "Open"
        elif count_open == 0 and grab_strength > 0.8:
            hand_status = "Closed"
        else:
            hand_status = "-"

        lines = [
            f"Hand: {hand_type}",
            f"Status: {hand_status}",
            "Fingers Openness:"
        ]
        for finger_name, openness in finger_openness:
            lines.append(f"  {finger_name}: {openness*100:.0f}%")

        for i, line in enumerate(lines):
            cv2.putText(
                self.output_image,
                line,
                (pos[0], pos[1] + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    def draw_finger_skeleton(self, hand):
        for index_digit in range(5):
            digit = hand.digits[index_digit]
            # finger_name = ["Thumb", "Index", "Middle", "Ring", "Pinky"][index_digit]

            for index_bone in range(4):
                bone = digit.bones[index_bone]
                # bone_name = ["Metacarpal", "Proximal", "Intermediate", "Distal"][index_bone]

                # if finger_name == "Thumb":
                #     # Print 3D joint coordinates
                #     print(f"{finger_name} - {bone_name}:")
                #     print(f"  Start(prev_joint) = ({bone.prev_joint.x:.2f}, {bone.prev_joint.y:.2f}, {bone.prev_joint.z:.2f})")
                #     print(f"  End  (next_joint) = ({bone.next_joint.x:.2f}, {bone.next_joint.y:.2f}, {bone.next_joint.z:.2f})")
                #     print("-" * 40)

                if self.hands_format == "Dots":
                    prev_joint = self.get_joint_position(bone.prev_joint)
                    next_joint = self.get_joint_position(bone.next_joint)
                    if prev_joint:
                        cv2.circle(self.output_image, prev_joint, 2, self.hands_colour, -1)
                    if next_joint:
                        cv2.circle(self.output_image, next_joint, 2, self.hands_colour, -1)

                if self.hands_format == "Skeleton":
                    wrist = self.get_joint_position(hand.arm.next_joint)
                    elbow = self.get_joint_position(hand.arm.prev_joint)
                    if wrist:
                        cv2.circle(self.output_image, wrist, 3, self.hands_colour, -1)
                    if elbow:
                        cv2.circle(self.output_image, elbow, 3, self.hands_colour, -1)
                    if wrist and elbow:
                        cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)

                    bone_start = self.get_joint_position(bone.prev_joint)
                    bone_end = self.get_joint_position(bone.next_joint)
                    if bone_start:
                        cv2.circle(self.output_image, bone_start, 3, self.hands_colour, -1)
                    if bone_end:
                        cv2.circle(self.output_image, bone_end, 3, self.hands_colour, -1)
                    if bone_start and bone_end:
                        cv2.line(self.output_image, bone_start, bone_end, self.hands_colour, 2)

                    if ((index_digit == 0) and (index_bone == 0)) or (
                        (index_digit > 0) and (index_digit < 4) and (index_bone < 2)
                    ):
                        index_digit_next = index_digit + 1
                        digit_next = hand.digits[index_digit_next]
                        bone_next = digit_next.bones[index_bone]
                        bone_next_start = self.get_joint_position(bone_next.prev_joint)
                        if bone_start and bone_next_start:
                            cv2.line(
                                self.output_image,
                                bone_start,
                                bone_next_start,
                                self.hands_colour,
                                2,
                            )
                    if index_bone == 0 and bone_start and wrist:
                        cv2.line(self.output_image, bone_start, wrist, self.hands_colour, 2)

    def render_hands(self, event):
        self.output_image[:, :] = 0
        center_x = self.screen_size[1] // 2
        center_y = self.screen_size[0] // 2
        cv2.circle(self.output_image, (center_x, center_y), 5, (0, 255, 255), -1)

        cv2.putText(
            self.output_image,
            f"Tracking Mode: {_TRACKING_MODES[self.tracking_mode]}",
            (10, self.screen_size[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.font_colour,
            1,
        )

        if len(event.hands) == 0:
            return

        # Optional: set starting Y position for finger info for each hand to avoid overlap
        finger_info_y = 20

        # for idx, hand in enumerate(event.hands):
        #     self.draw_palm(hand)
        #     # Draw finger info on left side with vertical offset per hand to avoid overlap
        #     self.draw_finger_info(hand, pos=(10, finger_info_y + idx * 80))
        #     self.draw_finger_skeleton(hand)
        #     self.send_finger_openness_uart(hand)

        right_hands = [hand for hand in event.hands if hand.type == leap.HandType.Right]

        if len(right_hands) == 0:
            return  # No right hand detected, skip drawing

        for idx, hand in enumerate(right_hands):
            self.draw_palm(hand)
            self.draw_finger_info(hand, pos=(10, finger_info_y + idx * 80))
            self.draw_finger_skeleton(hand)
            self.send_finger_openness_uart(hand)



class TrackingListener(leap.Listener):
    def __init__(self, canvas):
        self.canvas = canvas

    def on_connection_event(self, event):
        pass

    def on_tracking_mode_event(self, event):
        self.canvas.set_tracking_mode(event.current_tracking_mode)
        print(f"Tracking mode changed to {_TRACKING_MODES[event.current_tracking_mode]}")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        self.canvas.render_hands(event)


def main():
    # Setup UART
    try:
        ser = serial.Serial('COM6', 115200, timeout=1)  
        print(f"[INFO] Connected to STM32 on {ser.port}")
    except serial.SerialException as e:
        print(f"[ERROR] Cannot connect to STM32: {e}")
        ser = None

    canvas = Canvas(ser)

    print(canvas.name)
    print("")
    print("Press <key> in visualiser window to:")
    print("  x: Exit")
    print("  h: Select HMD tracking mode")
    print("  s: Select ScreenTop tracking mode")
    print("  d: Select Desktop tracking mode")
    print("  f: Toggle hands format between Skeleton/Dots")

    tracking_listener = TrackingListener(canvas)

    connection = leap.Connection()
    connection.add_listener(tracking_listener)

    running = True

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        canvas.set_tracking_mode(leap.TrackingMode.Desktop)

        while running:
            cv2.imshow(canvas.name, canvas.output_image)

            key = cv2.waitKey(1)

            if key == ord("x"):
                break
            elif key == ord("h"):
                connection.set_tracking_mode(leap.TrackingMode.HMD)
            elif key == ord("s"):
                connection.set_tracking_mode(leap.TrackingMode.ScreenTop)
            elif key == ord("d"):
                connection.set_tracking_mode(leap.TrackingMode.Desktop)
            elif key == ord("f"):
                canvas.toggle_hands_format()

    if ser and ser.is_open:
        ser.close()
        print("[INFO] Serial connection closed.")


main()
