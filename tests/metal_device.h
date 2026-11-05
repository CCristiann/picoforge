/* metal_device.h — see metal_device.m. */
#ifndef PICOFORGE_TEST_METAL_DEVICE_H
#define PICOFORGE_TEST_METAL_DEVICE_H

typedef struct MetalContext MetalContext;

MetalContext *metal_init(const char *metallib_path);
void         *metal_device(MetalContext *ctx);
void          metal_shutdown(MetalContext *ctx);

#endif
