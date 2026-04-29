const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub fn slice(from: []const u8, comptime from_type: types.DType, to: []u8, comptime to_type: types.DType) void {
    const from_bits = types.dtypeBits(from_type);
    const to_bits = types.dtypeBits(to_type);
    const count = (from.len * 8) / from_bits;
    std.debug.assert((to.len * 8) / to_bits >= count);
    c.nk_cast(from.ptr, @intFromEnum(from_type), count, to.ptr, @intFromEnum(to_type));
}

fn unary(comptime Source: type, comptime Target: type, comptime func: anytype, value: Source) Target {
    var src = value;
    var dst: Target = undefined;
    func(@ptrCast(&src), @ptrCast(&dst));
    return dst;
}

pub fn Scalar(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f16 => types.F16,
        .bf16 => types.BF16,
        .e4m3 => types.E4M3,
        .e5m2 => types.E5M2,
        .e2m3 => types.E2M3,
        .e3m2 => types.E3M2,
        else => @compileError("unsupported scalar cast dtype"),
    };
}

pub fn toF32(comptime dtype: types.DType, value: Scalar(dtype)) types.F32 {
    return switch (dtype) {
        .f16 => unary(types.F16, types.F32, c.nk_f16_to_f32, value),
        .bf16 => unary(types.BF16, types.F32, c.nk_bf16_to_f32, value),
        .e4m3 => unary(types.E4M3, types.F32, c.nk_e4m3_to_f32, value),
        .e5m2 => unary(types.E5M2, types.F32, c.nk_e5m2_to_f32, value),
        .e2m3 => unary(types.E2M3, types.F32, c.nk_e2m3_to_f32, value),
        .e3m2 => unary(types.E3M2, types.F32, c.nk_e3m2_to_f32, value),
        else => unreachable,
    };
}

pub fn fromF32(comptime dtype: types.DType, value: types.F32) Scalar(dtype) {
    return switch (dtype) {
        .f16 => unary(types.F32, types.F16, c.nk_f32_to_f16, value),
        .bf16 => unary(types.F32, types.BF16, c.nk_f32_to_bf16, value),
        .e4m3 => unary(types.F32, types.E4M3, c.nk_f32_to_e4m3, value),
        .e5m2 => unary(types.F32, types.E5M2, c.nk_f32_to_e5m2, value),
        .e2m3 => unary(types.F32, types.E2M3, c.nk_f32_to_e2m3, value),
        .e3m2 => unary(types.F32, types.E3M2, c.nk_f32_to_e3m2, value),
        else => unreachable,
    };
}

test "scalar casts round-trip representative values" {
    try std.testing.expectApproxEqAbs(@as(types.F32, 1.5), toF32(.f16, fromF32(.f16, 1.5)), 1e-3);
    try std.testing.expectApproxEqAbs(@as(types.F32, 2), toF32(.bf16, fromF32(.bf16, 2)), 1e-3);
    try std.testing.expectApproxEqAbs(@as(types.F32, 1), toF32(.e4m3, fromF32(.e4m3, 1)), 1e-3);
    try std.testing.expectApproxEqAbs(@as(types.F32, 1), toF32(.e5m2, fromF32(.e5m2, 1)), 1e-3);
    try std.testing.expectApproxEqAbs(@as(types.F32, 1), toF32(.e2m3, fromF32(.e2m3, 1)), 1e-3);
    try std.testing.expectApproxEqAbs(@as(types.F32, 1), toF32(.e3m2, fromF32(.e3m2, 1)), 1e-3);
}

test "slice cast writes caller-provided output" {
    const input = [_]types.F32{ 1.5, -2.25, 3.75 };
    var output = [_]types.F32{ 0, 0, 0 };
    slice(std.mem.sliceAsBytes(input[0..]), .f32, std.mem.sliceAsBytes(output[0..]), .f32);
    try std.testing.expectEqualSlices(types.F32, input[0..], output[0..]);
}
