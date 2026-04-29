const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub const Error = error{
    EmptyInput,
    InputTooSmall,
};

pub fn Storage(comptime dtype: types.DType) type {
    return switch (dtype) {
        .u1 => types.U1x8,
        .u8 => types.U8,
        .u16 => types.U16,
        .u32 => types.U32,
        else => @compileError("unsupported set dtype"),
    };
}

fn requiredBytes(comptime dtype: types.DType, count: usize) Error!usize {
    if (count == 0) return Error.EmptyInput;
    return switch (dtype) {
        .u1 => count / 8 + @intFromBool(count % 8 != 0),
        else => std.math.mul(usize, count, @sizeOf(Storage(dtype))) catch Error.InputTooSmall,
    };
}

fn expectStorage(comptime dtype: types.DType, a: []const u8, b: []const u8, count: usize) Error!void {
    const required = try requiredBytes(dtype, count);
    if (a.len < required or b.len < required) return Error.InputTooSmall;
}

pub fn hamming(comptime dtype: types.DType, a: []const u8, b: []const u8, count: usize) Error!types.U32 {
    try expectStorage(dtype, a, b, count);
    var result: types.U32 = undefined;
    switch (dtype) {
        .u1 => c.nk_hamming_u1(@ptrCast(@alignCast(a.ptr)), @ptrCast(@alignCast(b.ptr)), count, @ptrCast(&result)),
        .u8 => c.nk_hamming_u8(@ptrCast(@alignCast(a.ptr)), @ptrCast(@alignCast(b.ptr)), count, @ptrCast(&result)),
        else => @compileError("unsupported dtype for hamming"),
    }
    return result;
}

pub fn jaccard(comptime dtype: types.DType, a: []const u8, b: []const u8, count: usize) Error!types.F32 {
    try expectStorage(dtype, a, b, count);
    var result: types.F32 = undefined;
    switch (dtype) {
        .u1 => c.nk_jaccard_u1(@ptrCast(@alignCast(a.ptr)), @ptrCast(@alignCast(b.ptr)), count, &result),
        .u16 => c.nk_jaccard_u16(@ptrCast(@alignCast(a.ptr)), @ptrCast(@alignCast(b.ptr)), count, &result),
        .u32 => c.nk_jaccard_u32(@ptrCast(@alignCast(a.ptr)), @ptrCast(@alignCast(b.ptr)), count, &result),
        else => @compileError("unsupported dtype for jaccard"),
    }
    return result;
}

test "hamming distances" {
    const u8_a = [_]types.U8{ 1, 2, 3 };
    const u8_b = [_]types.U8{ 1, 9, 8 };
    try std.testing.expectEqual(@as(types.U32, 2), try hamming(.u8, std.mem.sliceAsBytes(u8_a[0..]), std.mem.sliceAsBytes(u8_b[0..]), u8_a.len));

    const u1_a = [_]types.U1x8{0b1010_1010};
    const u1_b = [_]types.U1x8{0b1111_0000};
    try std.testing.expectEqual(@as(types.U32, 4), try hamming(.u1, std.mem.sliceAsBytes(u1_a[0..]), std.mem.sliceAsBytes(u1_b[0..]), 8));
}

test "jaccard distances" {
    const u1_values = [_]types.U1x8{0b1010_1010};
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), try jaccard(.u1, std.mem.sliceAsBytes(u1_values[0..]), std.mem.sliceAsBytes(u1_values[0..]), 8), 1e-6);

    const u16_values = [_]types.U16{ 1, 2, 3 };
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), try jaccard(.u16, std.mem.sliceAsBytes(u16_values[0..]), std.mem.sliceAsBytes(u16_values[0..]), u16_values.len), 1e-6);

    const u32_values = [_]types.U32{ 1, 2, 3 };
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), try jaccard(.u32, std.mem.sliceAsBytes(u32_values[0..]), std.mem.sliceAsBytes(u32_values[0..]), u32_values.len), 1e-6);
}

test "set distances report invalid inputs" {
    const zero = [_]types.U8{0};
    try std.testing.expectError(Error.EmptyInput, hamming(.u8, std.mem.sliceAsBytes(zero[0..]), std.mem.sliceAsBytes(zero[0..]), 0));
    try std.testing.expectError(Error.InputTooSmall, hamming(.u8, std.mem.sliceAsBytes(zero[0..])[0..0], std.mem.sliceAsBytes(zero[0..]), 1));
}
