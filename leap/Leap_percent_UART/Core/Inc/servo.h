#ifndef SERVO_HANDLE_H
#define SERVO_HANDLE_H

#include "stm32f4xx_hal.h"  // Replace with your MCU's HAL

typedef struct
{
    TIM_HandleTypeDef* htim;   // Timer used for PWM
    uint32_t Channel;          // PWM channel (e.g., TIM_CHANNEL_1)
    uint16_t MinPulse;         // Minimum pulse width in microseconds (e.g., 500us)
    uint16_t MaxPulse;         // Maximum pulse width in microseconds (e.g., 2500us)
} Servo_HandleTypeDef;

// Initializes the servo handle with timer, channel, and pulse range
void Servo_Init(Servo_HandleTypeDef* hservo, TIM_HandleTypeDef* htim, uint32_t channel, uint16_t min_us, uint16_t max_us);

// Sets the servo angle (0–180 degrees)
void Servo_SetAngle(Servo_HandleTypeDef* hservo, uint8_t angle);

#endif // __SERVO_HANDLE_H
