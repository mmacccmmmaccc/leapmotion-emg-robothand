#ifndef NOTCH_FILTER_H
#define NOTCH_FILTER_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/*  Header-only multi-channel IIR Notch (RBJ biquad, DF-II Transposed)
    API (integer-literal friendly):
        NotchState s;
        Notch_Init(&s, 1000, 50, 30);                 // fs=1000 Hz, f0=50 Hz, Q=30
        Notch_FilterArray(&s, adc_values, filt, 6, 0, 4095);
*/

typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float *z1;
    float *z2;
    size_t capacity;
} NotchState;

/* ---- internal helpers ---- */
static inline int _notch_reserve(NotchState* s, size_t needed) {
    if (needed <= s->capacity) return 1;
    float* new_z1 = (float*)realloc(s->z1, needed * sizeof(float));
    if (!new_z1) return 0;
    float* new_z2 = (float*)realloc(s->z2, needed * sizeof(float));
    if (!new_z2) { free(new_z1); return 0; }
    // zero-init newly added portion
    if (s->capacity < needed) {
        size_t add = needed - s->capacity;
        memset(new_z1 + s->capacity, 0, add * sizeof(float));
        memset(new_z2 + s->capacity, 0, add * sizeof(float));
    }
    s->z1 = new_z1;
    s->z2 = new_z2;
    s->capacity = needed;
    return 1;
}

static inline void _notch_set_coeffs_f32(NotchState* s, float fs_hz, float f0_hz, float Q) {
    // RBJ cookbook notch
    const float w0    = 2.0f * (float)M_PI * (f0_hz / fs_hz);
    const float cosw0 = cosf(w0);
    const float sinw0 = sinf(w0);
    const float alpha = sinw0 / (2.0f * Q);

    const float a0 = 1.0f + alpha;
    const float b0 = 1.0f;
    const float b1 = -2.0f * cosw0;
    const float b2 = 1.0f;
    const float a1 = -2.0f * cosw0;
    const float a2 = 1.0f - alpha;

    s->b0 = b0 / a0;
    s->b1 = b1 / a0;
    s->b2 = b2 / a0;
    s->a1 = a1 / a0;
    s->a2 = a2 / a0;
}

/* ---- public API ---- */

/* Accepts ints/doubles; internally casts to float */
static inline void Notch_Init(NotchState* s, double fs_hz, double f0_hz, double Q) {
    s->z1 = NULL;
    s->z2 = NULL;
    s->capacity = 0;
    _notch_set_coeffs_f32(s, (float)fs_hz, (float)f0_hz, (float)Q);
}

/* Change coefficients later (e.g., 50 ↔ 60 Hz) */
static inline void Notch_SetCoeffs(NotchState* s, double fs_hz, double f0_hz, double Q) {
    _notch_set_coeffs_f32(s, (float)fs_hz, (float)f0_hz, (float)Q);
}

/* Optional cleanup */
static inline void Notch_Deinit(NotchState* s) {
    free(s->z1); s->z1 = NULL;
    free(s->z2); s->z2 = NULL;
    s->capacity = 0;
}

/**
 * Filter a frame (one sample per channel).
 * @param s         filter state
 * @param in        input codes (N channels)
 * @param out       output codes (N channels)
 * @param channels  N
 * @param min_code  clamp min (plain int OK)
 * @param max_code  clamp max (plain int OK)
 */
static inline void Notch_FilterArray(NotchState* s,
                                     const uint16_t* in,
                                     uint16_t* out,
                                     size_t channels,
                                     int min_code,
                                     int max_code)
{
    // Sanitize and pre-cast once
    const uint16_t umin = (min_code < 0) ? 0u : (uint16_t)min_code;
    const uint16_t umax = (max_code < 0) ? 0u : (uint16_t)max_code;

    if (!_notch_reserve(s, channels)) {
        // allocation failed → passthrough
        for (size_t i = 0; i < channels; ++i) out[i] = in[i];
        return;
    }

    const float b0 = s->b0, b1 = s->b1, b2 = s->b2, a1 = s->a1, a2 = s->a2;

    for (size_t ch = 0; ch < channels; ++ch) {
        const float x = (float)in[ch];
        float y = b0 * x + s->z1[ch];
        s->z1[ch] = b1 * x + s->z2[ch] - a1 * y;
        s->z2[ch] = b2 * x            - a2 * y;

        if (y < (float)umin) y = (float)umin;
        if (y > (float)umax) y = (float)umax;
        out[ch] = (uint16_t)(y + 0.5f);
    }
}

#endif /* NOTCH_FILTER_H */
