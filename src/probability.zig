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
        .f16, .bf16 => types.F32,
        else => @compileError("unsupported dtype for probability divergence"),
    };
}

fn requiredBytes(comptime A: type, count: usize) Error!usize {
    if (count == 0) return Error.EmptyInput;
    return std.math.mul(usize, count, @sizeOf(A)) catch Error.InputTooSmall;
}

fn divergenceImpl(comptime A: type, comptime R: type, comptime func: anytype, a: []const u8, b: []const u8, count: usize) Error!R {
    const required = try requiredBytes(A, count);
    if (a.len < required or b.len < required) return Error.InputTooSmall;
    var result: R = undefined;
    func(@ptrCast(@alignCast(a.ptr)), @ptrCast(@alignCast(b.ptr)), count, @ptrCast(&result));
    return result;
}

pub fn kld(comptime dtype: types.DType, a: []const u8, b: []const u8, count: usize) Error!Result(dtype) {
    return switch (dtype) {
        .f64 => divergenceImpl(types.F64, types.F64, c.nk_kld_f64, a, b, count),
        .f32 => divergenceImpl(types.F32, types.F64, c.nk_kld_f32, a, b, count),
        .f16 => divergenceImpl(types.F16, types.F32, c.nk_kld_f16, a, b, count),
        .bf16 => divergenceImpl(types.BF16, types.F32, c.nk_kld_bf16, a, b, count),
        else => @compileError("unsupported dtype for kld"),
    };
}

pub fn jsd(comptime dtype: types.DType, a: []const u8, b: []const u8, count: usize) Error!Result(dtype) {
    return switch (dtype) {
        .f64 => divergenceImpl(types.F64, types.F64, c.nk_jsd_f64, a, b, count),
        .f32 => divergenceImpl(types.F32, types.F64, c.nk_jsd_f32, a, b, count),
        .f16 => divergenceImpl(types.F16, types.F32, c.nk_jsd_f16, a, b, count),
        .bf16 => divergenceImpl(types.BF16, types.F32, c.nk_jsd_bf16, a, b, count),
        else => @compileError("unsupported dtype for jsd"),
    };
}

fn expectZero(comptime dtype: types.DType, data: []const u8, count: usize) !void {
    const kld_result = try kld(dtype, data, data, count);
    const jsd_result = try jsd(dtype, data, data, count);
    switch (@TypeOf(kld_result)) {
        types.F64 => {
            try std.testing.expectApproxEqAbs(@as(types.F64, 0), kld_result, 1e-12);
            try std.testing.expectApproxEqAbs(@as(types.F64, 0), jsd_result, 1e-12);
        },
        types.F32 => {
            try std.testing.expectApproxEqAbs(@as(types.F32, 0), kld_result, 1e-6);
            try std.testing.expectApproxEqAbs(@as(types.F32, 0), jsd_result, 1e-6);
        },
        else => unreachable,
    }
}

test "probability comptime dtype API returns exact f32 result" {
    const p = [_]types.F32{ 0.25, 0.75 };
    const bytes = std.mem.sliceAsBytes(p[0..]);
    const result = try kld(.f32, bytes, bytes, p.len);
    try std.testing.expectEqual(types.F64, @TypeOf(result));
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), result, 1e-6);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), try jsd(.f32, bytes, bytes, p.len), 1e-6);
}

test "probability comptime dtype API covers supported dtypes" {
    const cast = @import("cast.zig");
    const p_f64 = [_]types.F64{ 0.25, 0.75 };
    const p_f16 = [_]types.F16{ cast.fromF32(.f16, 0.25), cast.fromF32(.f16, 0.75) };
    const p_bf16 = [_]types.BF16{ cast.fromF32(.bf16, 0.25), cast.fromF32(.bf16, 0.75) };

    try expectZero(.f64, std.mem.sliceAsBytes(p_f64[0..]), p_f64.len);
    try expectZero(.f16, std.mem.sliceAsBytes(p_f16[0..]), p_f16.len);
    try expectZero(.bf16, std.mem.sliceAsBytes(p_bf16[0..]), p_bf16.len);
}

test "probability comptime dtype API reports invalid inputs" {
    const p = [_]types.F32{1};
    try std.testing.expectError(Error.EmptyInput, kld(.f32, std.mem.sliceAsBytes(p[0..]), std.mem.sliceAsBytes(p[0..]), 0));
    try std.testing.expectError(Error.InputTooSmall, jsd(.f32, std.mem.sliceAsBytes(p[0..])[0..3], std.mem.sliceAsBytes(p[0..]), 1));
}
