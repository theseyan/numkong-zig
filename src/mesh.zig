const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub const Error = error{
    EmptyInput,
    InputTooSmall,
};

pub fn Transform(comptime T: type, comptime R: type) type {
    return struct {
        a_centroid: [3]T,
        b_centroid: [3]T,
        rotation: [9]T,
        scale: T,
        rmsd: R,
    };
}

pub fn Result(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f64 => Transform(types.F64, types.F64),
        .f32 => Transform(types.F32, types.F64),
        .f16, .bf16 => Transform(types.F32, types.F32),
        else => @compileError("unsupported dtype for mesh transform"),
    };
}

fn requiredBytes(comptime A: type, point_count: usize) Error!usize {
    if (point_count == 0) return Error.EmptyInput;
    const coordinate_count = std.math.mul(usize, point_count, 3) catch return Error.InputTooSmall;
    return std.math.mul(usize, coordinate_count, @sizeOf(A)) catch Error.InputTooSmall;
}

fn transformImpl(comptime A: type, comptime T: type, comptime R: type, comptime func: anytype, a: []const u8, b: []const u8, point_count: usize) Error!Transform(T, R) {
    const required = try requiredBytes(A, point_count);
    if (a.len < required or b.len < required) return Error.InputTooSmall;
    var result: Transform(T, R) = undefined;
    func(@ptrCast(@alignCast(a.ptr)), @ptrCast(@alignCast(b.ptr)), point_count, @ptrCast(&result.a_centroid), @ptrCast(&result.b_centroid), @ptrCast(&result.rotation), @ptrCast(&result.scale), @ptrCast(&result.rmsd));
    return result;
}

pub fn rmsd(comptime dtype: types.DType, a: []const u8, b: []const u8, point_count: usize) Error!Result(dtype) {
    return switch (dtype) {
        .f64 => transformImpl(types.F64, types.F64, types.F64, c.nk_rmsd_f64, a, b, point_count),
        .f32 => transformImpl(types.F32, types.F32, types.F64, c.nk_rmsd_f32, a, b, point_count),
        .f16 => transformImpl(types.F16, types.F32, types.F32, c.nk_rmsd_f16, a, b, point_count),
        .bf16 => transformImpl(types.BF16, types.F32, types.F32, c.nk_rmsd_bf16, a, b, point_count),
        else => @compileError("unsupported dtype for rmsd"),
    };
}

pub fn kabsch(comptime dtype: types.DType, a: []const u8, b: []const u8, point_count: usize) Error!Result(dtype) {
    return switch (dtype) {
        .f64 => transformImpl(types.F64, types.F64, types.F64, c.nk_kabsch_f64, a, b, point_count),
        .f32 => transformImpl(types.F32, types.F32, types.F64, c.nk_kabsch_f32, a, b, point_count),
        .f16 => transformImpl(types.F16, types.F32, types.F32, c.nk_kabsch_f16, a, b, point_count),
        .bf16 => transformImpl(types.BF16, types.F32, types.F32, c.nk_kabsch_bf16, a, b, point_count),
        else => @compileError("unsupported dtype for kabsch"),
    };
}

pub fn umeyama(comptime dtype: types.DType, a: []const u8, b: []const u8, point_count: usize) Error!Result(dtype) {
    return switch (dtype) {
        .f64 => transformImpl(types.F64, types.F64, types.F64, c.nk_umeyama_f64, a, b, point_count),
        .f32 => transformImpl(types.F32, types.F32, types.F64, c.nk_umeyama_f32, a, b, point_count),
        .f16 => transformImpl(types.F16, types.F32, types.F32, c.nk_umeyama_f16, a, b, point_count),
        .bf16 => transformImpl(types.BF16, types.F32, types.F32, c.nk_umeyama_bf16, a, b, point_count),
        else => @compileError("unsupported dtype for umeyama"),
    };
}

fn expectZero(comptime dtype: types.DType, data: []const u8, point_count: usize, tolerance: anytype) !void {
    const rmsd_result = try rmsd(dtype, data, data, point_count);
    const kabsch_result = try kabsch(dtype, data, data, point_count);
    const umeyama_result = try umeyama(dtype, data, data, point_count);
    const zero: @TypeOf(rmsd_result.rmsd) = 0;
    try std.testing.expectApproxEqAbs(zero, rmsd_result.rmsd, tolerance);
    try std.testing.expectApproxEqAbs(zero, kabsch_result.rmsd, tolerance);
    try std.testing.expectApproxEqAbs(zero, umeyama_result.rmsd, tolerance);
}

test "mesh comptime dtype API returns exact f32 result" {
    const points32 = [_]types.F32{
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
    };
    const bytes = std.mem.sliceAsBytes(points32[0..]);
    const result = try rmsd(.f32, bytes, bytes, 3);
    try std.testing.expectEqual(Result(.f32), @TypeOf(result));
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), result.rmsd, 1e-6);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), (try kabsch(.f32, bytes, bytes, 3)).rmsd, 1e-6);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), (try umeyama(.f32, bytes, bytes, 3)).rmsd, 1e-6);
}

test "mesh comptime dtype API covers supported dtypes" {
    const cast = @import("cast.zig");
    const points64 = [_]types.F64{
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
    };
    try expectZero(.f64, std.mem.sliceAsBytes(points64[0..]), 3, 1e-12);

    const points16 = [_]types.F16{
        cast.fromF32(.f16, 0), cast.fromF32(.f16, 0), cast.fromF32(.f16, 0),
        cast.fromF32(.f16, 1), cast.fromF32(.f16, 0), cast.fromF32(.f16, 0),
        cast.fromF32(.f16, 0), cast.fromF32(.f16, 1), cast.fromF32(.f16, 0),
    };
    try expectZero(.f16, std.mem.sliceAsBytes(points16[0..]), 3, 1e-3);

    const points_bf16 = [_]types.BF16{
        cast.fromF32(.bf16, 0), cast.fromF32(.bf16, 0), cast.fromF32(.bf16, 0),
        cast.fromF32(.bf16, 1), cast.fromF32(.bf16, 0), cast.fromF32(.bf16, 0),
        cast.fromF32(.bf16, 0), cast.fromF32(.bf16, 1), cast.fromF32(.bf16, 0),
    };
    try expectZero(.bf16, std.mem.sliceAsBytes(points_bf16[0..]), 3, 1e-3);
}

test "mesh comptime dtype API reports invalid inputs" {
    const point = [_]types.F32{ 0, 0, 0 };
    try std.testing.expectError(Error.EmptyInput, rmsd(.f32, std.mem.sliceAsBytes(point[0..]), std.mem.sliceAsBytes(point[0..]), 0));
    try std.testing.expectError(Error.InputTooSmall, kabsch(.f32, std.mem.sliceAsBytes(point[0..])[0..3], std.mem.sliceAsBytes(point[0..]), 1));
}
