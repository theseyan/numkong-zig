const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub const Error = error{
    InputLengthMismatch,
    OutputTooSmall,
};

fn ScalarType(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f64, .i32, .u32, .i64, .u64 => types.F64,
        .f32, .f16, .bf16, .e4m3, .e5m2, .e2m3, .e3m2, .i8, .u8, .i16, .u16 => types.F32,
        .f32c => types.F32C,
        .f64c => types.F64C,
        else => @compileError("unsupported dtype for element-wise affine operations"),
    };
}

fn countFor(comptime A: type, bytes: []const u8) Error!usize {
    if (bytes.len % @sizeOf(A) != 0) return Error.InputLengthMismatch;
    return bytes.len / @sizeOf(A);
}

fn expectSameLen(expected: usize, bytes: []const u8) Error!void {
    if (bytes.len != expected) return Error.InputLengthMismatch;
}

fn expectOutLen(expected: usize, bytes: []u8) Error!void {
    if (bytes.len < expected) return Error.OutputTooSmall;
}

fn scaleImpl(comptime A: type, comptime S: type, comptime func: anytype, a: []const u8, alpha: S, beta: S, out: []u8) Error!void {
    const count = try countFor(A, a);
    try expectOutLen(a.len, out);
    var alpha_var = alpha;
    var beta_var = beta;
    func(@ptrCast(@alignCast(a.ptr)), count, @ptrCast(&alpha_var), @ptrCast(&beta_var), @ptrCast(@alignCast(out.ptr)));
}

fn sumImpl(comptime A: type, comptime func: anytype, a: []const u8, b: []const u8, out: []u8) Error!void {
    const count = try countFor(A, a);
    try expectSameLen(a.len, b);
    try expectOutLen(a.len, out);
    func(@ptrCast(@alignCast(a.ptr)), @ptrCast(@alignCast(b.ptr)), count, @ptrCast(@alignCast(out.ptr)));
}

fn blendImpl(comptime A: type, comptime S: type, comptime func: anytype, a: []const u8, b: []const u8, alpha: S, beta: S, out: []u8) Error!void {
    const count = try countFor(A, a);
    try expectSameLen(a.len, b);
    try expectOutLen(a.len, out);
    var alpha_var = alpha;
    var beta_var = beta;
    func(@ptrCast(@alignCast(a.ptr)), @ptrCast(@alignCast(b.ptr)), count, @ptrCast(&alpha_var), @ptrCast(&beta_var), @ptrCast(@alignCast(out.ptr)));
}

fn fmaImpl(comptime A: type, comptime S: type, comptime func: anytype, a: []const u8, b: []const u8, d: []const u8, alpha: S, beta: S, out: []u8) Error!void {
    const count = try countFor(A, a);
    try expectSameLen(a.len, b);
    try expectSameLen(a.len, d);
    try expectOutLen(a.len, out);
    var alpha_var = alpha;
    var beta_var = beta;
    func(@ptrCast(@alignCast(a.ptr)), @ptrCast(@alignCast(b.ptr)), @ptrCast(@alignCast(d.ptr)), count, @ptrCast(&alpha_var), @ptrCast(&beta_var), @ptrCast(@alignCast(out.ptr)));
}

fn unaryImpl(comptime A: type, comptime func: anytype, a: []const u8, out: []u8) Error!void {
    const count = try countFor(A, a);
    try expectOutLen(a.len, out);
    func(@ptrCast(@alignCast(a.ptr)), count, @ptrCast(@alignCast(out.ptr)));
}

pub fn scale(comptime dtype: types.DType, a: []const u8, alpha: ScalarType(dtype), beta: ScalarType(dtype), out: []u8) Error!void {
    return switch (dtype) {
        .f64 => scaleImpl(types.F64, types.F64, c.nk_each_scale_f64, a, alpha, beta, out),
        .f32 => scaleImpl(types.F32, types.F32, c.nk_each_scale_f32, a, alpha, beta, out),
        .f16 => scaleImpl(types.F16, types.F32, c.nk_each_scale_f16, a, alpha, beta, out),
        .bf16 => scaleImpl(types.BF16, types.F32, c.nk_each_scale_bf16, a, alpha, beta, out),
        .e4m3 => scaleImpl(types.E4M3, types.F32, c.nk_each_scale_e4m3, a, alpha, beta, out),
        .e5m2 => scaleImpl(types.E5M2, types.F32, c.nk_each_scale_e5m2, a, alpha, beta, out),
        .e2m3 => scaleImpl(types.E2M3, types.F32, c.nk_each_scale_e2m3, a, alpha, beta, out),
        .e3m2 => scaleImpl(types.E3M2, types.F32, c.nk_each_scale_e3m2, a, alpha, beta, out),
        .i8 => scaleImpl(types.I8, types.F32, c.nk_each_scale_i8, a, alpha, beta, out),
        .u8 => scaleImpl(types.U8, types.F32, c.nk_each_scale_u8, a, alpha, beta, out),
        .i16 => scaleImpl(types.I16, types.F32, c.nk_each_scale_i16, a, alpha, beta, out),
        .u16 => scaleImpl(types.U16, types.F32, c.nk_each_scale_u16, a, alpha, beta, out),
        .i32 => scaleImpl(types.I32, types.F64, c.nk_each_scale_i32, a, alpha, beta, out),
        .u32 => scaleImpl(types.U32, types.F64, c.nk_each_scale_u32, a, alpha, beta, out),
        .i64 => scaleImpl(types.I64, types.F64, c.nk_each_scale_i64, a, alpha, beta, out),
        .u64 => scaleImpl(types.U64, types.F64, c.nk_each_scale_u64, a, alpha, beta, out),
        .f32c => scaleImpl(types.F32C, types.F32C, c.nk_each_scale_f32c, a, alpha, beta, out),
        .f64c => scaleImpl(types.F64C, types.F64C, c.nk_each_scale_f64c, a, alpha, beta, out),
        else => @compileError("unsupported dtype for scale"),
    };
}

pub fn sum(comptime dtype: types.DType, a: []const u8, b: []const u8, out: []u8) Error!void {
    return switch (dtype) {
        .f64 => sumImpl(types.F64, c.nk_each_sum_f64, a, b, out),
        .f32 => sumImpl(types.F32, c.nk_each_sum_f32, a, b, out),
        .f16 => sumImpl(types.F16, c.nk_each_sum_f16, a, b, out),
        .bf16 => sumImpl(types.BF16, c.nk_each_sum_bf16, a, b, out),
        .e4m3 => sumImpl(types.E4M3, c.nk_each_sum_e4m3, a, b, out),
        .e5m2 => sumImpl(types.E5M2, c.nk_each_sum_e5m2, a, b, out),
        .e2m3 => sumImpl(types.E2M3, c.nk_each_sum_e2m3, a, b, out),
        .e3m2 => sumImpl(types.E3M2, c.nk_each_sum_e3m2, a, b, out),
        .i8 => sumImpl(types.I8, c.nk_each_sum_i8, a, b, out),
        .u8 => sumImpl(types.U8, c.nk_each_sum_u8, a, b, out),
        .i16 => sumImpl(types.I16, c.nk_each_sum_i16, a, b, out),
        .u16 => sumImpl(types.U16, c.nk_each_sum_u16, a, b, out),
        .i32 => sumImpl(types.I32, c.nk_each_sum_i32, a, b, out),
        .u32 => sumImpl(types.U32, c.nk_each_sum_u32, a, b, out),
        .i64 => sumImpl(types.I64, c.nk_each_sum_i64, a, b, out),
        .u64 => sumImpl(types.U64, c.nk_each_sum_u64, a, b, out),
        .f32c => sumImpl(types.F32C, c.nk_each_sum_f32c, a, b, out),
        .f64c => sumImpl(types.F64C, c.nk_each_sum_f64c, a, b, out),
        else => @compileError("unsupported dtype for sum"),
    };
}

pub fn blend(comptime dtype: types.DType, a: []const u8, b: []const u8, alpha: ScalarType(dtype), beta: ScalarType(dtype), out: []u8) Error!void {
    return switch (dtype) {
        .f64 => blendImpl(types.F64, types.F64, c.nk_each_blend_f64, a, b, alpha, beta, out),
        .f32 => blendImpl(types.F32, types.F32, c.nk_each_blend_f32, a, b, alpha, beta, out),
        .f16 => blendImpl(types.F16, types.F32, c.nk_each_blend_f16, a, b, alpha, beta, out),
        .bf16 => blendImpl(types.BF16, types.F32, c.nk_each_blend_bf16, a, b, alpha, beta, out),
        .e4m3 => blendImpl(types.E4M3, types.F32, c.nk_each_blend_e4m3, a, b, alpha, beta, out),
        .e5m2 => blendImpl(types.E5M2, types.F32, c.nk_each_blend_e5m2, a, b, alpha, beta, out),
        .e2m3 => blendImpl(types.E2M3, types.F32, c.nk_each_blend_e2m3, a, b, alpha, beta, out),
        .e3m2 => blendImpl(types.E3M2, types.F32, c.nk_each_blend_e3m2, a, b, alpha, beta, out),
        .i8 => blendImpl(types.I8, types.F32, c.nk_each_blend_i8, a, b, alpha, beta, out),
        .u8 => blendImpl(types.U8, types.F32, c.nk_each_blend_u8, a, b, alpha, beta, out),
        .i16 => blendImpl(types.I16, types.F32, c.nk_each_blend_i16, a, b, alpha, beta, out),
        .u16 => blendImpl(types.U16, types.F32, c.nk_each_blend_u16, a, b, alpha, beta, out),
        .i32 => blendImpl(types.I32, types.F64, c.nk_each_blend_i32, a, b, alpha, beta, out),
        .u32 => blendImpl(types.U32, types.F64, c.nk_each_blend_u32, a, b, alpha, beta, out),
        .i64 => blendImpl(types.I64, types.F64, c.nk_each_blend_i64, a, b, alpha, beta, out),
        .u64 => blendImpl(types.U64, types.F64, c.nk_each_blend_u64, a, b, alpha, beta, out),
        .f32c => blendImpl(types.F32C, types.F32C, c.nk_each_blend_f32c, a, b, alpha, beta, out),
        .f64c => blendImpl(types.F64C, types.F64C, c.nk_each_blend_f64c, a, b, alpha, beta, out),
        else => @compileError("unsupported dtype for blend"),
    };
}

pub fn fma(comptime dtype: types.DType, a: []const u8, b: []const u8, d: []const u8, alpha: ScalarType(dtype), beta: ScalarType(dtype), out: []u8) Error!void {
    return switch (dtype) {
        .f64 => fmaImpl(types.F64, types.F64, c.nk_each_fma_f64, a, b, d, alpha, beta, out),
        .f32 => fmaImpl(types.F32, types.F32, c.nk_each_fma_f32, a, b, d, alpha, beta, out),
        .f16 => fmaImpl(types.F16, types.F32, c.nk_each_fma_f16, a, b, d, alpha, beta, out),
        .bf16 => fmaImpl(types.BF16, types.F32, c.nk_each_fma_bf16, a, b, d, alpha, beta, out),
        .e4m3 => fmaImpl(types.E4M3, types.F32, c.nk_each_fma_e4m3, a, b, d, alpha, beta, out),
        .e5m2 => fmaImpl(types.E5M2, types.F32, c.nk_each_fma_e5m2, a, b, d, alpha, beta, out),
        .e2m3 => fmaImpl(types.E2M3, types.F32, c.nk_each_fma_e2m3, a, b, d, alpha, beta, out),
        .e3m2 => fmaImpl(types.E3M2, types.F32, c.nk_each_fma_e3m2, a, b, d, alpha, beta, out),
        .i8 => fmaImpl(types.I8, types.F32, c.nk_each_fma_i8, a, b, d, alpha, beta, out),
        .u8 => fmaImpl(types.U8, types.F32, c.nk_each_fma_u8, a, b, d, alpha, beta, out),
        .i16 => fmaImpl(types.I16, types.F32, c.nk_each_fma_i16, a, b, d, alpha, beta, out),
        .u16 => fmaImpl(types.U16, types.F32, c.nk_each_fma_u16, a, b, d, alpha, beta, out),
        .i32 => fmaImpl(types.I32, types.F64, c.nk_each_fma_i32, a, b, d, alpha, beta, out),
        .u32 => fmaImpl(types.U32, types.F64, c.nk_each_fma_u32, a, b, d, alpha, beta, out),
        .i64 => fmaImpl(types.I64, types.F64, c.nk_each_fma_i64, a, b, d, alpha, beta, out),
        .u64 => fmaImpl(types.U64, types.F64, c.nk_each_fma_u64, a, b, d, alpha, beta, out),
        .f32c => fmaImpl(types.F32C, types.F32C, c.nk_each_fma_f32c, a, b, d, alpha, beta, out),
        .f64c => fmaImpl(types.F64C, types.F64C, c.nk_each_fma_f64c, a, b, d, alpha, beta, out),
        else => @compileError("unsupported dtype for fma"),
    };
}

pub fn sin(comptime dtype: types.DType, a: []const u8, out: []u8) Error!void {
    return switch (dtype) {
        .f64 => unaryImpl(types.F64, c.nk_each_sin_f64, a, out),
        .f32 => unaryImpl(types.F32, c.nk_each_sin_f32, a, out),
        .f16 => unaryImpl(types.F16, c.nk_each_sin_f16, a, out),
        else => @compileError("unsupported dtype for sin"),
    };
}

pub fn cos(comptime dtype: types.DType, a: []const u8, out: []u8) Error!void {
    return switch (dtype) {
        .f64 => unaryImpl(types.F64, c.nk_each_cos_f64, a, out),
        .f32 => unaryImpl(types.F32, c.nk_each_cos_f32, a, out),
        .f16 => unaryImpl(types.F16, c.nk_each_cos_f16, a, out),
        else => @compileError("unsupported dtype for cos"),
    };
}

pub fn atan(comptime dtype: types.DType, a: []const u8, out: []u8) Error!void {
    return switch (dtype) {
        .f64 => unaryImpl(types.F64, c.nk_each_atan_f64, a, out),
        .f32 => unaryImpl(types.F32, c.nk_each_atan_f32, a, out),
        .f16 => unaryImpl(types.F16, c.nk_each_atan_f16, a, out),
        else => @compileError("unsupported dtype for atan"),
    };
}

fn exerciseArithmetic(comptime dtype: types.DType, data: []const u8, one: ScalarType(dtype), zero: ScalarType(dtype), out: []u8) !void {
    try scale(dtype, data, one, zero, out);
    try sum(dtype, data, data, out);
    try blend(dtype, data, data, one, zero, out);
    try fma(dtype, data, data, data, one, zero, out);
}

test "each arithmetic byte API writes caller output" {
    const xs_f32 = [_]types.F32{ 1, 2 };
    const ys_f32 = [_]types.F32{ 3, 4 };
    var out_f32 = [_]types.F32{ 0, 0 };
    const x_bytes = std.mem.sliceAsBytes(xs_f32[0..]);
    const y_bytes = std.mem.sliceAsBytes(ys_f32[0..]);
    const out_bytes = std.mem.sliceAsBytes(out_f32[0..]);

    try scale(.f32, x_bytes, 2, 1, out_bytes);
    try std.testing.expectEqualSlices(types.F32, &.{ 3, 5 }, &out_f32);
    try sum(.f32, x_bytes, y_bytes, out_bytes);
    try std.testing.expectEqualSlices(types.F32, &.{ 4, 6 }, &out_f32);
    try blend(.f32, x_bytes, y_bytes, 2, 3, out_bytes);
    try std.testing.expectEqualSlices(types.F32, &.{ 11, 16 }, &out_f32);
    try fma(.f32, x_bytes, y_bytes, x_bytes, 2, 3, out_bytes);
    try std.testing.expectEqualSlices(types.F32, &.{ 9, 22 }, &out_f32);

    const xs_f64 = [_]types.F64{ 1, 2 };
    var out_f64 = [_]types.F64{ 0, 0 };
    try exerciseArithmetic(.f64, std.mem.sliceAsBytes(xs_f64[0..]), 1, 0, std.mem.sliceAsBytes(out_f64[0..]));
    try std.testing.expectEqual(@as(types.F64, 1), out_f64[0]);
}

test "each arithmetic byte API covers all dtypes" {
    const cast = @import("cast.zig");
    const zero_f16 = [_]types.F16{cast.fromF32(.f16, 0)};
    const zero_bf16 = [_]types.BF16{cast.fromF32(.bf16, 0)};
    const zero_e4m3 = [_]types.E4M3{cast.fromF32(.e4m3, 0)};
    const zero_e5m2 = [_]types.E5M2{cast.fromF32(.e5m2, 0)};
    const zero_e2m3 = [_]types.E2M3{cast.fromF32(.e2m3, 0)};
    const zero_e3m2 = [_]types.E3M2{cast.fromF32(.e3m2, 0)};
    const zero_i8 = [_]types.I8{0};
    const zero_u8 = [_]types.U8{0};
    const zero_i16 = [_]types.I16{0};
    const zero_u16 = [_]types.U16{0};
    const zero_i32 = [_]types.I32{0};
    const zero_u32 = [_]types.U32{0};
    const zero_i64 = [_]types.I64{0};
    const zero_u64 = [_]types.U64{0};
    const zero_f32c = [_]types.F32C{.{ .real = 0, .imag = 0 }};
    const zero_f64c = [_]types.F64C{.{ .real = 0, .imag = 0 }};

    var out_f16 = zero_f16;
    var out_bf16 = zero_bf16;
    var out_e4m3 = zero_e4m3;
    var out_e5m2 = zero_e5m2;
    var out_e2m3 = zero_e2m3;
    var out_e3m2 = zero_e3m2;
    var out_i8 = zero_i8;
    var out_u8 = zero_u8;
    var out_i16 = zero_i16;
    var out_u16 = zero_u16;
    var out_i32 = zero_i32;
    var out_u32 = zero_u32;
    var out_i64 = zero_i64;
    var out_u64 = zero_u64;
    var out_f32c = zero_f32c;
    var out_f64c = zero_f64c;

    const one_f32: types.F32 = 1;
    const zero_scalar_f32: types.F32 = 0;
    const one_f64: types.F64 = 1;
    const zero_scalar_f64: types.F64 = 0;
    const one_f32c = types.F32C{ .real = 1, .imag = 0 };
    const zero_scalar_f32c = types.F32C{ .real = 0, .imag = 0 };
    const one_f64c = types.F64C{ .real = 1, .imag = 0 };
    const zero_scalar_f64c = types.F64C{ .real = 0, .imag = 0 };

    try exerciseArithmetic(.f16, std.mem.sliceAsBytes(zero_f16[0..]), one_f32, zero_scalar_f32, std.mem.sliceAsBytes(out_f16[0..]));
    try exerciseArithmetic(.bf16, std.mem.sliceAsBytes(zero_bf16[0..]), one_f32, zero_scalar_f32, std.mem.sliceAsBytes(out_bf16[0..]));
    try exerciseArithmetic(.e4m3, std.mem.sliceAsBytes(zero_e4m3[0..]), one_f32, zero_scalar_f32, std.mem.sliceAsBytes(out_e4m3[0..]));
    try exerciseArithmetic(.e5m2, std.mem.sliceAsBytes(zero_e5m2[0..]), one_f32, zero_scalar_f32, std.mem.sliceAsBytes(out_e5m2[0..]));
    try exerciseArithmetic(.e2m3, std.mem.sliceAsBytes(zero_e2m3[0..]), one_f32, zero_scalar_f32, std.mem.sliceAsBytes(out_e2m3[0..]));
    try exerciseArithmetic(.e3m2, std.mem.sliceAsBytes(zero_e3m2[0..]), one_f32, zero_scalar_f32, std.mem.sliceAsBytes(out_e3m2[0..]));
    try exerciseArithmetic(.i8, std.mem.sliceAsBytes(zero_i8[0..]), one_f32, zero_scalar_f32, std.mem.sliceAsBytes(out_i8[0..]));
    try exerciseArithmetic(.u8, std.mem.sliceAsBytes(zero_u8[0..]), one_f32, zero_scalar_f32, std.mem.sliceAsBytes(out_u8[0..]));
    try exerciseArithmetic(.i16, std.mem.sliceAsBytes(zero_i16[0..]), one_f32, zero_scalar_f32, std.mem.sliceAsBytes(out_i16[0..]));
    try exerciseArithmetic(.u16, std.mem.sliceAsBytes(zero_u16[0..]), one_f32, zero_scalar_f32, std.mem.sliceAsBytes(out_u16[0..]));
    try exerciseArithmetic(.i32, std.mem.sliceAsBytes(zero_i32[0..]), one_f64, zero_scalar_f64, std.mem.sliceAsBytes(out_i32[0..]));
    try exerciseArithmetic(.u32, std.mem.sliceAsBytes(zero_u32[0..]), one_f64, zero_scalar_f64, std.mem.sliceAsBytes(out_u32[0..]));
    try exerciseArithmetic(.i64, std.mem.sliceAsBytes(zero_i64[0..]), one_f64, zero_scalar_f64, std.mem.sliceAsBytes(out_i64[0..]));
    try exerciseArithmetic(.u64, std.mem.sliceAsBytes(zero_u64[0..]), one_f64, zero_scalar_f64, std.mem.sliceAsBytes(out_u64[0..]));
    try exerciseArithmetic(.f32c, std.mem.sliceAsBytes(zero_f32c[0..]), one_f32c, zero_scalar_f32c, std.mem.sliceAsBytes(out_f32c[0..]));
    try exerciseArithmetic(.f64c, std.mem.sliceAsBytes(zero_f64c[0..]), one_f64c, zero_scalar_f64c, std.mem.sliceAsBytes(out_f64c[0..]));

    try std.testing.expectApproxEqAbs(@as(types.F32, 0), cast.toF32(.f16, out_f16[0]), 1e-6);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), cast.toF32(.bf16, out_bf16[0]), 1e-6);
    try std.testing.expectEqual(@as(types.I8, 0), out_i8[0]);
    try std.testing.expectEqual(@as(types.U64, 0), out_u64[0]);
    try std.testing.expectEqual(@as(types.F32, 0), out_f32c[0].real);
    try std.testing.expectEqual(@as(types.F64, 0), out_f64c[0].imag);
}

test "each trigonometry byte API covers supported dtypes" {
    const cast = @import("cast.zig");
    const zero_f64 = [_]types.F64{0};
    const zero_f32 = [_]types.F32{0};
    const zero_f16 = [_]types.F16{cast.fromF32(.f16, 0)};
    var out_f64 = [_]types.F64{0};
    var out_f32 = [_]types.F32{0};
    var out_f16 = [_]types.F16{cast.fromF32(.f16, 0)};
    const f64_in = std.mem.sliceAsBytes(zero_f64[0..]);
    const f32_in = std.mem.sliceAsBytes(zero_f32[0..]);
    const f16_in = std.mem.sliceAsBytes(zero_f16[0..]);
    const f64_out = std.mem.sliceAsBytes(out_f64[0..]);
    const f32_out = std.mem.sliceAsBytes(out_f32[0..]);
    const f16_out = std.mem.sliceAsBytes(out_f16[0..]);

    try sin(.f64, f64_in, f64_out);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), out_f64[0], 1e-12);
    try cos(.f64, f64_in, f64_out);
    try std.testing.expectApproxEqAbs(@as(types.F64, 1), out_f64[0], 1e-12);
    try atan(.f64, f64_in, f64_out);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), out_f64[0], 1e-12);

    try sin(.f32, f32_in, f32_out);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), out_f32[0], 1e-6);
    try cos(.f32, f32_in, f32_out);
    try std.testing.expectApproxEqAbs(@as(types.F32, 1), out_f32[0], 1e-6);
    try atan(.f32, f32_in, f32_out);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), out_f32[0], 1e-6);

    try sin(.f16, f16_in, f16_out);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), cast.toF32(.f16, out_f16[0]), 1e-3);
    try cos(.f16, f16_in, f16_out);
    try std.testing.expectApproxEqAbs(@as(types.F32, 1), cast.toF32(.f16, out_f16[0]), 1e-3);
    try atan(.f16, f16_in, f16_out);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), cast.toF32(.f16, out_f16[0]), 1e-3);
}

test "each byte API reports invalid inputs" {
    var out = [_]types.F32{0};
    try std.testing.expectError(Error.InputLengthMismatch, sum(.f32, std.mem.asBytes(&out)[0..3], std.mem.asBytes(&out), std.mem.asBytes(&out)));
}
