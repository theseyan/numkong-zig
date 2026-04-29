const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub const Error = error{
    EmptyInput,
    InputTooSmall,
};

pub fn EuclideanResult(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f64, .f32 => types.F64,
        .f16, .bf16, .e4m3, .e5m2, .e2m3, .e3m2, .i8, .u8, .i4, .u4 => types.F32,
        else => @compileError("unsupported dtype for euclidean"),
    };
}

pub fn SqEuclideanResult(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f64, .f32 => types.F64,
        .f16, .bf16, .e4m3, .e5m2, .e2m3, .e3m2 => types.F32,
        .i8, .u8, .i4, .u4 => types.U32,
        else => @compileError("unsupported dtype for squared euclidean"),
    };
}

pub fn AngularResult(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f64, .f32 => types.F64,
        .f16, .bf16, .e4m3, .e5m2, .e2m3, .e3m2, .i8, .u8, .i4, .u4 => types.F32,
        else => @compileError("unsupported dtype for angular"),
    };
}

fn requiredBytes(comptime A: type, count: usize, comptime values_per_byte: usize) Error!usize {
    if (count == 0) return Error.EmptyInput;
    if (values_per_byte == 1) return std.math.mul(usize, count, @sizeOf(A)) catch Error.InputTooSmall;
    return count / values_per_byte + @intFromBool(count % values_per_byte != 0);
}

fn expectStorage(comptime A: type, a: []const u8, b: []const u8, count: usize, comptime values_per_byte: usize) Error!void {
    const required = try requiredBytes(A, count, values_per_byte);
    if (a.len < required or b.len < required) return Error.InputTooSmall;
}

fn metricImpl(comptime A: type, comptime R: type, comptime func: anytype, a: []const u8, b: []const u8, count: usize, comptime values_per_byte: usize) Error!R {
    try expectStorage(A, a, b, count, values_per_byte);
    var result: R = undefined;
    func(@ptrCast(@alignCast(a.ptr)), @ptrCast(@alignCast(b.ptr)), count, @ptrCast(&result));
    return result;
}

pub fn euclidean(comptime dtype: types.DType, a: []const u8, b: []const u8, count: usize) Error!EuclideanResult(dtype) {
    return switch (dtype) {
        .f64 => metricImpl(types.F64, types.F64, c.nk_euclidean_f64, a, b, count, 1),
        .f32 => metricImpl(types.F32, types.F64, c.nk_euclidean_f32, a, b, count, 1),
        .f16 => metricImpl(types.F16, types.F32, c.nk_euclidean_f16, a, b, count, 1),
        .bf16 => metricImpl(types.BF16, types.F32, c.nk_euclidean_bf16, a, b, count, 1),
        .e4m3 => metricImpl(types.E4M3, types.F32, c.nk_euclidean_e4m3, a, b, count, 1),
        .e5m2 => metricImpl(types.E5M2, types.F32, c.nk_euclidean_e5m2, a, b, count, 1),
        .e2m3 => metricImpl(types.E2M3, types.F32, c.nk_euclidean_e2m3, a, b, count, 1),
        .e3m2 => metricImpl(types.E3M2, types.F32, c.nk_euclidean_e3m2, a, b, count, 1),
        .i8 => metricImpl(types.I8, types.F32, c.nk_euclidean_i8, a, b, count, 1),
        .u8 => metricImpl(types.U8, types.F32, c.nk_euclidean_u8, a, b, count, 1),
        .i4 => metricImpl(types.I4x2, types.F32, c.nk_euclidean_i4, a, b, count, 2),
        .u4 => metricImpl(types.U4x2, types.F32, c.nk_euclidean_u4, a, b, count, 2),
        else => @compileError("unsupported dtype for euclidean"),
    };
}

pub fn sqeuclidean(comptime dtype: types.DType, a: []const u8, b: []const u8, count: usize) Error!SqEuclideanResult(dtype) {
    return switch (dtype) {
        .f64 => metricImpl(types.F64, types.F64, c.nk_sqeuclidean_f64, a, b, count, 1),
        .f32 => metricImpl(types.F32, types.F64, c.nk_sqeuclidean_f32, a, b, count, 1),
        .f16 => metricImpl(types.F16, types.F32, c.nk_sqeuclidean_f16, a, b, count, 1),
        .bf16 => metricImpl(types.BF16, types.F32, c.nk_sqeuclidean_bf16, a, b, count, 1),
        .e4m3 => metricImpl(types.E4M3, types.F32, c.nk_sqeuclidean_e4m3, a, b, count, 1),
        .e5m2 => metricImpl(types.E5M2, types.F32, c.nk_sqeuclidean_e5m2, a, b, count, 1),
        .e2m3 => metricImpl(types.E2M3, types.F32, c.nk_sqeuclidean_e2m3, a, b, count, 1),
        .e3m2 => metricImpl(types.E3M2, types.F32, c.nk_sqeuclidean_e3m2, a, b, count, 1),
        .i8 => metricImpl(types.I8, types.U32, c.nk_sqeuclidean_i8, a, b, count, 1),
        .u8 => metricImpl(types.U8, types.U32, c.nk_sqeuclidean_u8, a, b, count, 1),
        .i4 => metricImpl(types.I4x2, types.U32, c.nk_sqeuclidean_i4, a, b, count, 2),
        .u4 => metricImpl(types.U4x2, types.U32, c.nk_sqeuclidean_u4, a, b, count, 2),
        else => @compileError("unsupported dtype for squared euclidean"),
    };
}

pub fn angular(comptime dtype: types.DType, a: []const u8, b: []const u8, count: usize) Error!AngularResult(dtype) {
    return switch (dtype) {
        .f64 => metricImpl(types.F64, types.F64, c.nk_angular_f64, a, b, count, 1),
        .f32 => metricImpl(types.F32, types.F64, c.nk_angular_f32, a, b, count, 1),
        .f16 => metricImpl(types.F16, types.F32, c.nk_angular_f16, a, b, count, 1),
        .bf16 => metricImpl(types.BF16, types.F32, c.nk_angular_bf16, a, b, count, 1),
        .e4m3 => metricImpl(types.E4M3, types.F32, c.nk_angular_e4m3, a, b, count, 1),
        .e5m2 => metricImpl(types.E5M2, types.F32, c.nk_angular_e5m2, a, b, count, 1),
        .e2m3 => metricImpl(types.E2M3, types.F32, c.nk_angular_e2m3, a, b, count, 1),
        .e3m2 => metricImpl(types.E3M2, types.F32, c.nk_angular_e3m2, a, b, count, 1),
        .i8 => metricImpl(types.I8, types.F32, c.nk_angular_i8, a, b, count, 1),
        .u8 => metricImpl(types.U8, types.F32, c.nk_angular_u8, a, b, count, 1),
        .i4 => metricImpl(types.I4x2, types.F32, c.nk_angular_i4, a, b, count, 2),
        .u4 => metricImpl(types.U4x2, types.F32, c.nk_angular_u4, a, b, count, 2),
        else => @compileError("unsupported dtype for angular"),
    };
}

fn expectZero(comptime dtype: types.DType, data: []const u8, count: usize) !void {
    const euclidean_result = try euclidean(dtype, data, data, count);
    const angular_result = try angular(dtype, data, data, count);
    switch (@TypeOf(euclidean_result)) {
        types.F64 => try std.testing.expectApproxEqAbs(@as(types.F64, 0), euclidean_result, 1e-12),
        types.F32 => try std.testing.expectApproxEqAbs(@as(types.F32, 0), euclidean_result, 1e-6),
        else => unreachable,
    }
    switch (@TypeOf(angular_result)) {
        types.F64 => try std.testing.expectApproxEqAbs(@as(types.F64, 0), angular_result, 1e-12),
        types.F32 => try std.testing.expectApproxEqAbs(@as(types.F32, 0), angular_result, 1e-6),
        else => unreachable,
    }

    const sqeuclidean_result = try sqeuclidean(dtype, data, data, count);
    switch (@TypeOf(sqeuclidean_result)) {
        types.F64 => try std.testing.expectApproxEqAbs(@as(types.F64, 0), sqeuclidean_result, 1e-12),
        types.F32 => try std.testing.expectApproxEqAbs(@as(types.F32, 0), sqeuclidean_result, 1e-6),
        types.U32 => try std.testing.expectEqual(@as(types.U32, 0), sqeuclidean_result),
        else => unreachable,
    }
}

test "spatial comptime dtype API returns exact f32 results" {
    const a = [_]types.F32{ 0, 0 };
    const b = [_]types.F32{ 3, 4 };
    const a_bytes = std.mem.sliceAsBytes(a[0..]);
    const b_bytes = std.mem.sliceAsBytes(b[0..]);
    const euclidean_result = try euclidean(.f32, a_bytes, b_bytes, a.len);
    try std.testing.expectEqual(types.F64, @TypeOf(euclidean_result));
    try std.testing.expectApproxEqAbs(@as(types.F64, 5), euclidean_result, 1e-6);
    try std.testing.expectApproxEqAbs(@as(types.F64, 25), try sqeuclidean(.f32, a_bytes, b_bytes, a.len), 1e-6);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), try angular(.f32, b_bytes, b_bytes, b.len), 1e-6);
}

test "spatial comptime dtype API covers all dtypes" {
    const cast = @import("cast.zig");
    const zero_f64 = [_]types.F64{0};
    const zero_f16 = [_]types.F16{cast.fromF32(.f16, 0)};
    const zero_bf16 = [_]types.BF16{cast.fromF32(.bf16, 0)};
    const zero_e4m3 = [_]types.E4M3{cast.fromF32(.e4m3, 0)};
    const zero_e5m2 = [_]types.E5M2{cast.fromF32(.e5m2, 0)};
    const zero_e2m3 = [_]types.E2M3{cast.fromF32(.e2m3, 0)};
    const zero_e3m2 = [_]types.E3M2{cast.fromF32(.e3m2, 0)};
    const zero_i8 = [_]types.I8{0};
    const zero_u8 = [_]types.U8{0};
    const zero_i4 = [_]types.I4x2{0};
    const zero_u4 = [_]types.U4x2{0};

    try expectZero(.f64, std.mem.sliceAsBytes(zero_f64[0..]), 1);
    try expectZero(.f16, std.mem.sliceAsBytes(zero_f16[0..]), 1);
    try expectZero(.bf16, std.mem.sliceAsBytes(zero_bf16[0..]), 1);
    try expectZero(.e4m3, std.mem.sliceAsBytes(zero_e4m3[0..]), 1);
    try expectZero(.e5m2, std.mem.sliceAsBytes(zero_e5m2[0..]), 1);
    try expectZero(.e2m3, std.mem.sliceAsBytes(zero_e2m3[0..]), 1);
    try expectZero(.e3m2, std.mem.sliceAsBytes(zero_e3m2[0..]), 1);
    try expectZero(.i8, std.mem.sliceAsBytes(zero_i8[0..]), 1);
    try expectZero(.u8, std.mem.sliceAsBytes(zero_u8[0..]), 1);
    try expectZero(.i4, std.mem.sliceAsBytes(zero_i4[0..]), 2);
    try expectZero(.u4, std.mem.sliceAsBytes(zero_u4[0..]), 2);
}

test "spatial comptime dtype API reports invalid inputs" {
    const zero = [_]types.F32{0};
    try std.testing.expectError(Error.EmptyInput, euclidean(.f32, std.mem.sliceAsBytes(zero[0..]), std.mem.sliceAsBytes(zero[0..]), 0));
    try std.testing.expectError(Error.InputTooSmall, sqeuclidean(.f32, std.mem.sliceAsBytes(zero[0..])[0..3], std.mem.sliceAsBytes(zero[0..]), 1));
}
