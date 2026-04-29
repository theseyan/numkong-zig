const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub fn Float(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f32 => types.F32,
        .f64 => types.F64,
        else => @compileError("unsupported floating scalar dtype"),
    };
}

pub fn Integer(comptime dtype: types.DType) type {
    return switch (dtype) {
        .u8 => types.U8,
        .i8 => types.I8,
        .u16 => types.U16,
        .i16 => types.I16,
        .u32 => types.U32,
        .i32 => types.I32,
        .u64 => types.U64,
        .i64 => types.I64,
        else => @compileError("unsupported integer scalar dtype"),
    };
}

pub fn LowPrecision(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f16 => types.F16,
        .bf16 => types.BF16,
        .e4m3 => types.E4M3,
        .e5m2 => types.E5M2,
        .e2m3 => types.E2M3,
        .e3m2 => types.E3M2,
        else => @compileError("unsupported low-precision scalar dtype"),
    };
}

pub fn sqrt(comptime dtype: types.DType, x: Float(dtype)) Float(dtype) {
    return switch (dtype) {
        .f32 => c.nk_f32_sqrt(x),
        .f64 => c.nk_f64_sqrt(x),
        else => unreachable,
    };
}

pub fn rsqrt(comptime dtype: types.DType, x: Float(dtype)) Float(dtype) {
    return switch (dtype) {
        .f32 => c.nk_f32_rsqrt(x),
        .f64 => c.nk_f64_rsqrt(x),
        else => unreachable,
    };
}

pub fn fma(comptime dtype: types.DType, a: Float(dtype), b: Float(dtype), addend: Float(dtype)) Float(dtype) {
    return switch (dtype) {
        .f32 => c.nk_f32_fma(a, b, addend),
        .f64 => c.nk_f64_fma(a, b, addend),
        else => unreachable,
    };
}

pub fn saturatingAdd(comptime dtype: types.DType, a: Integer(dtype), b: Integer(dtype)) Integer(dtype) {
    return switch (dtype) {
        .u8 => c.nk_u8_saturating_add(a, b),
        .i8 => c.nk_i8_saturating_add(a, b),
        .u16 => @intCast(c.nk_u16_saturating_add(a, b)),
        .i16 => @intCast(c.nk_i16_saturating_add(a, b)),
        .u32 => @intCast(c.nk_u32_saturating_add(a, b)),
        .i32 => @intCast(c.nk_i32_saturating_add(a, b)),
        .u64 => @intCast(c.nk_u64_saturating_add(a, b)),
        .i64 => @intCast(c.nk_i64_saturating_add(a, b)),
        else => unreachable,
    };
}

pub fn saturatingMul(comptime dtype: types.DType, a: Integer(dtype), b: Integer(dtype)) Integer(dtype) {
    return switch (dtype) {
        .u8 => c.nk_u8_saturating_mul(a, b),
        .i8 => c.nk_i8_saturating_mul(a, b),
        .u16 => @intCast(c.nk_u16_saturating_mul(a, b)),
        .i16 => @intCast(c.nk_i16_saturating_mul(a, b)),
        .u32 => @intCast(c.nk_u32_saturating_mul(a, b)),
        .i32 => @intCast(c.nk_i32_saturating_mul(a, b)),
        .u64 => @intCast(c.nk_u64_saturating_mul(a, b)),
        .i64 => @intCast(c.nk_i64_saturating_mul(a, b)),
        else => unreachable,
    };
}

pub fn order(comptime dtype: types.DType, a: LowPrecision(dtype), b: LowPrecision(dtype)) i32 {
    return @intCast(switch (dtype) {
        .f16 => c.nk_f16_order(a, b),
        .bf16 => c.nk_bf16_order(a, b),
        .e4m3 => c.nk_e4m3_order(a, b),
        .e5m2 => c.nk_e5m2_order(a, b),
        .e2m3 => c.nk_e2m3_order(a, b),
        .e3m2 => c.nk_e3m2_order(a, b),
        else => unreachable,
    });
}

test "floating point scalar helpers" {
    try std.testing.expectApproxEqAbs(@as(types.F32, 3), sqrt(.f32, 9), 1e-6);
    try std.testing.expectApproxEqAbs(@as(types.F64, 4), sqrt(.f64, 16), 1e-12);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0.25), rsqrt(.f32, 16), 1e-6);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0.2), rsqrt(.f64, 25), 1e-12);
    try std.testing.expectApproxEqAbs(@as(types.F32, 10), fma(.f32, 2, 3, 4), 1e-6);
    try std.testing.expectApproxEqAbs(@as(types.F64, 10), fma(.f64, 2, 3, 4), 1e-12);
}

test "saturating integer arithmetic" {
    try std.testing.expectEqual(@as(types.U8, 255), saturatingAdd(.u8, 250, 10));
    try std.testing.expectEqual(@as(types.I8, 127), saturatingAdd(.i8, 120, 20));
    try std.testing.expectEqual(@as(types.I8, -128), saturatingAdd(.i8, -120, -20));
    try std.testing.expectEqual(@as(types.U16, 65535), saturatingMul(.u16, 400, 400));
    try std.testing.expectEqual(@as(types.I16, 32767), saturatingMul(.i16, 400, 400));
    try std.testing.expectEqual(@as(types.I16, -32768), saturatingMul(.i16, -400, 400));
    try std.testing.expectEqual(@as(types.U32, std.math.maxInt(types.U32)), saturatingAdd(.u32, std.math.maxInt(types.U32), 1));
    try std.testing.expectEqual(@as(types.I32, std.math.minInt(types.I32)), saturatingMul(.i32, std.math.minInt(types.I32), 2));
    try std.testing.expectEqual(@as(types.U64, std.math.maxInt(types.U64)), saturatingMul(.u64, std.math.maxInt(types.U64), 2));
    try std.testing.expectEqual(@as(types.I64, std.math.maxInt(types.I64)), saturatingAdd(.i64, std.math.maxInt(types.I64), 1));
}

test "low precision ordering" {
    const cast = @import("cast.zig");
    const zero_f16 = cast.fromF32(.f16, 0);
    const one_f16 = cast.fromF32(.f16, 1);
    const zero_bf16 = cast.fromF32(.bf16, 0);
    const one_bf16 = cast.fromF32(.bf16, 1);
    const zero_e4m3 = cast.fromF32(.e4m3, 0);
    const one_e4m3 = cast.fromF32(.e4m3, 1);
    const zero_e5m2 = cast.fromF32(.e5m2, 0);
    const one_e5m2 = cast.fromF32(.e5m2, 1);
    const zero_e2m3 = cast.fromF32(.e2m3, 0);
    const one_e2m3 = cast.fromF32(.e2m3, 1);
    const zero_e3m2 = cast.fromF32(.e3m2, 0);
    const one_e3m2 = cast.fromF32(.e3m2, 1);

    try std.testing.expect(order(.f16, zero_f16, one_f16) < 0);
    try std.testing.expectEqual(@as(i32, 0), order(.f16, one_f16, one_f16));
    try std.testing.expect(order(.bf16, one_bf16, zero_bf16) > 0);
    try std.testing.expect(order(.e4m3, zero_e4m3, one_e4m3) < 0);
    try std.testing.expect(order(.e5m2, zero_e5m2, one_e5m2) < 0);
    try std.testing.expect(order(.e2m3, zero_e2m3, one_e2m3) < 0);
    try std.testing.expect(order(.e3m2, zero_e3m2, one_e3m2) < 0);
}
