const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub const Error = error{
    EmptyInput,
    InputTooSmall,
};

pub fn Result(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f64, .f32 => types.F64,
        .f16, .bf16, .e4m3, .e5m2, .e2m3, .e3m2 => types.F32,
        .i8, .i4 => types.I32,
        .u8, .u4, .u1 => types.U32,
        .f64c, .f32c => types.F64C,
        .f16c, .bf16c => types.F32C,
        else => @compileError("unsupported dtype for dot"),
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

fn dotImpl(comptime A: type, comptime R: type, comptime func: anytype, a: []const u8, b: []const u8, count: usize, comptime values_per_byte: usize) Error!R {
    try expectStorage(A, a, b, count, values_per_byte);
    var result: R = undefined;
    func(@ptrCast(@alignCast(a.ptr)), @ptrCast(@alignCast(b.ptr)), count, @ptrCast(&result));
    return result;
}

pub fn dot(comptime dtype: types.DType, a: []const u8, b: []const u8, count: usize) Error!Result(dtype) {
    return switch (dtype) {
        .f64 => dotImpl(types.F64, types.F64, c.nk_dot_f64, a, b, count, 1),
        .f32 => dotImpl(types.F32, types.F64, c.nk_dot_f32, a, b, count, 1),
        .f16 => dotImpl(types.F16, types.F32, c.nk_dot_f16, a, b, count, 1),
        .bf16 => dotImpl(types.BF16, types.F32, c.nk_dot_bf16, a, b, count, 1),
        .i8 => dotImpl(types.I8, types.I32, c.nk_dot_i8, a, b, count, 1),
        .u8 => dotImpl(types.U8, types.U32, c.nk_dot_u8, a, b, count, 1),
        .i4 => dotImpl(types.I4x2, types.I32, c.nk_dot_i4, a, b, count, 2),
        .u4 => dotImpl(types.U4x2, types.U32, c.nk_dot_u4, a, b, count, 2),
        .u1 => dotImpl(types.U1x8, types.U32, c.nk_dot_u1, a, b, count, 8),
        .e4m3 => dotImpl(types.E4M3, types.F32, c.nk_dot_e4m3, a, b, count, 1),
        .e5m2 => dotImpl(types.E5M2, types.F32, c.nk_dot_e5m2, a, b, count, 1),
        .e2m3 => dotImpl(types.E2M3, types.F32, c.nk_dot_e2m3, a, b, count, 1),
        .e3m2 => dotImpl(types.E3M2, types.F32, c.nk_dot_e3m2, a, b, count, 1),
        .f64c => dotImpl(types.F64C, types.F64C, c.nk_dot_f64c, a, b, count, 1),
        .f32c => dotImpl(types.F32C, types.F64C, c.nk_dot_f32c, a, b, count, 1),
        .f16c => dotImpl(types.F16C, types.F32C, c.nk_dot_f16c, a, b, count, 1),
        .bf16c => dotImpl(types.BF16C, types.F32C, c.nk_dot_bf16c, a, b, count, 1),
        else => @compileError("unsupported dtype for dot"),
    };
}

pub fn vdot(comptime dtype: types.DType, a: []const u8, b: []const u8, count: usize) Error!Result(dtype) {
    return switch (dtype) {
        .f64c => dotImpl(types.F64C, types.F64C, c.nk_vdot_f64c, a, b, count, 1),
        .f32c => dotImpl(types.F32C, types.F64C, c.nk_vdot_f32c, a, b, count, 1),
        .f16c => dotImpl(types.F16C, types.F32C, c.nk_vdot_f16c, a, b, count, 1),
        .bf16c => dotImpl(types.BF16C, types.F32C, c.nk_vdot_bf16c, a, b, count, 1),
        else => @compileError("unsupported dtype for vdot"),
    };
}

fn expectZero(comptime dtype: types.DType, data: []const u8, count: usize) !void {
    const result = try dot(dtype, data, data, count);
    switch (@TypeOf(result)) {
        types.F64 => try std.testing.expectApproxEqAbs(@as(types.F64, 0), result, 1e-12),
        types.F32 => try std.testing.expectApproxEqAbs(@as(types.F32, 0), result, 1e-6),
        types.I32 => try std.testing.expectEqual(@as(types.I32, 0), result),
        types.U32 => try std.testing.expectEqual(@as(types.U32, 0), result),
        types.F64C => {
            try std.testing.expectApproxEqAbs(@as(types.F64, 0), result.real, 1e-12);
            try std.testing.expectApproxEqAbs(@as(types.F64, 0), result.imag, 1e-12);
        },
        types.F32C => {
            try std.testing.expectApproxEqAbs(@as(types.F32, 0), result.real, 1e-6);
            try std.testing.expectApproxEqAbs(@as(types.F32, 0), result.imag, 1e-6);
        },
        else => unreachable,
    }
}

test "dot comptime dtype API returns exact real results" {
    const a_f32 = [_]types.F32{ 1, 2, 3 };
    const b_f32 = [_]types.F32{ 4, 5, 6 };
    const result = try dot(.f32, std.mem.sliceAsBytes(a_f32[0..]), std.mem.sliceAsBytes(b_f32[0..]), a_f32.len);
    try std.testing.expectEqual(types.F64, @TypeOf(result));
    try std.testing.expectEqual(@as(types.F64, 32), result);

    const a_i8 = [_]types.I8{ -1, 2, 3 };
    const b_i8 = [_]types.I8{ 4, 5, -6 };
    try std.testing.expectEqual(@as(types.I32, -12), try dot(.i8, std.mem.sliceAsBytes(a_i8[0..]), std.mem.sliceAsBytes(b_i8[0..]), a_i8.len));
}

test "dot comptime dtype API covers real and packed dtypes" {
    const cast = @import("cast.zig");
    const zero_f64 = [_]types.F64{0};
    const zero_f16 = [_]types.F16{cast.fromF32(.f16, 0)};
    const zero_bf16 = [_]types.BF16{cast.fromF32(.bf16, 0)};
    const zero_e4m3 = [_]types.E4M3{cast.fromF32(.e4m3, 0)};
    const zero_e5m2 = [_]types.E5M2{cast.fromF32(.e5m2, 0)};
    const zero_e2m3 = [_]types.E2M3{cast.fromF32(.e2m3, 0)};
    const zero_e3m2 = [_]types.E3M2{cast.fromF32(.e3m2, 0)};
    const zero_u8 = [_]types.U8{0};
    const zero_i4 = [_]types.I4x2{0};
    const zero_u4 = [_]types.U4x2{0};
    const zero_u1 = [_]types.U1x8{0};

    try expectZero(.f64, std.mem.sliceAsBytes(zero_f64[0..]), 1);
    try expectZero(.f16, std.mem.sliceAsBytes(zero_f16[0..]), 1);
    try expectZero(.bf16, std.mem.sliceAsBytes(zero_bf16[0..]), 1);
    try expectZero(.e4m3, std.mem.sliceAsBytes(zero_e4m3[0..]), 1);
    try expectZero(.e5m2, std.mem.sliceAsBytes(zero_e5m2[0..]), 1);
    try expectZero(.e2m3, std.mem.sliceAsBytes(zero_e2m3[0..]), 1);
    try expectZero(.e3m2, std.mem.sliceAsBytes(zero_e3m2[0..]), 1);
    try expectZero(.u8, std.mem.sliceAsBytes(zero_u8[0..]), 1);
    try expectZero(.i4, std.mem.sliceAsBytes(zero_i4[0..]), 2);
    try expectZero(.u4, std.mem.sliceAsBytes(zero_u4[0..]), 2);
    try expectZero(.u1, std.mem.sliceAsBytes(zero_u1[0..]), 8);
}

test "dot comptime dtype API covers complex dtypes" {
    const cast = @import("cast.zig");
    const a_f32c = [_]types.F32C{.{ .real = 1, .imag = 2 }};
    const b_f32c = [_]types.F32C{.{ .real = 3, .imag = 4 }};
    const a_f64c = [_]types.F64C{.{ .real = 1, .imag = 2 }};
    const b_f64c = [_]types.F64C{.{ .real = 3, .imag = 4 }};
    const zero_f16c = [_]types.F16C{.{ .real = cast.fromF32(.f16, 0), .imag = cast.fromF32(.f16, 0) }};
    const zero_bf16c = [_]types.BF16C{.{ .real = cast.fromF32(.bf16, 0), .imag = cast.fromF32(.bf16, 0) }};

    const dot_f32c = try dot(.f32c, std.mem.sliceAsBytes(a_f32c[0..]), std.mem.sliceAsBytes(b_f32c[0..]), 1);
    try std.testing.expectApproxEqAbs(@as(types.F64, -5), dot_f32c.real, 1e-6);
    try std.testing.expectApproxEqAbs(@as(types.F64, 10), dot_f32c.imag, 1e-6);
    const vdot_f32c = try vdot(.f32c, std.mem.sliceAsBytes(a_f32c[0..]), std.mem.sliceAsBytes(b_f32c[0..]), 1);
    try std.testing.expectApproxEqAbs(@as(types.F64, 11), vdot_f32c.real, 1e-6);
    try std.testing.expectApproxEqAbs(@as(types.F64, -2), vdot_f32c.imag, 1e-6);

    const dot_f64c = try dot(.f64c, std.mem.sliceAsBytes(a_f64c[0..]), std.mem.sliceAsBytes(b_f64c[0..]), 1);
    try std.testing.expectApproxEqAbs(@as(types.F64, -5), dot_f64c.real, 1e-12);
    try std.testing.expectApproxEqAbs(@as(types.F64, 10), dot_f64c.imag, 1e-12);
    const vdot_f64c = try vdot(.f64c, std.mem.sliceAsBytes(a_f64c[0..]), std.mem.sliceAsBytes(b_f64c[0..]), 1);
    try std.testing.expectApproxEqAbs(@as(types.F64, 11), vdot_f64c.real, 1e-12);
    try std.testing.expectApproxEqAbs(@as(types.F64, -2), vdot_f64c.imag, 1e-12);

    try expectZero(.f16c, std.mem.sliceAsBytes(zero_f16c[0..]), 1);
    try expectZero(.bf16c, std.mem.sliceAsBytes(zero_bf16c[0..]), 1);
    const vdot_f16c = try vdot(.f16c, std.mem.sliceAsBytes(zero_f16c[0..]), std.mem.sliceAsBytes(zero_f16c[0..]), 1);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), vdot_f16c.real, 1e-6);
    const vdot_bf16c = try vdot(.bf16c, std.mem.sliceAsBytes(zero_bf16c[0..]), std.mem.sliceAsBytes(zero_bf16c[0..]), 1);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), vdot_bf16c.imag, 1e-6);
}

test "dot comptime dtype API reports invalid inputs" {
    const zero = [_]types.F32{0};
    try std.testing.expectError(Error.EmptyInput, dot(.f32, std.mem.sliceAsBytes(zero[0..]), std.mem.sliceAsBytes(zero[0..]), 0));
    try std.testing.expectError(Error.InputTooSmall, dot(.f32, std.mem.sliceAsBytes(zero[0..])[0..3], std.mem.sliceAsBytes(zero[0..]), 1));
}
