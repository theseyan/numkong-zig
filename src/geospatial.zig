const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub const Error = error{
    EmptyInput,
    InputTooSmall,
    OutputTooSmall,
};

pub fn Coordinate(comptime dtype: types.DType) type {
    return switch (dtype) {
        .f64 => types.F64,
        .f32 => types.F32,
        else => @compileError("unsupported dtype for geospatial distance"),
    };
}

fn requiredBytes(comptime T: type, count: usize) Error!usize {
    if (count == 0) return Error.EmptyInput;
    return std.math.mul(usize, count, @sizeOf(T)) catch Error.InputTooSmall;
}

fn distanceImpl(comptime T: type, comptime func: anytype, a_lats: []const u8, a_lons: []const u8, b_lats: []const u8, b_lons: []const u8, out: []u8, count: usize) Error!void {
    const required = try requiredBytes(T, count);
    if (a_lats.len < required or a_lons.len < required or b_lats.len < required or b_lons.len < required) return Error.InputTooSmall;
    if (out.len < required) return Error.OutputTooSmall;
    func(@ptrCast(@alignCast(a_lats.ptr)), @ptrCast(@alignCast(a_lons.ptr)), @ptrCast(@alignCast(b_lats.ptr)), @ptrCast(@alignCast(b_lons.ptr)), count, @ptrCast(@alignCast(out.ptr)));
}

pub fn haversine(comptime dtype: types.DType, a_lats: []const u8, a_lons: []const u8, b_lats: []const u8, b_lons: []const u8, out: []u8, count: usize) Error!void {
    return switch (dtype) {
        .f64 => distanceImpl(types.F64, c.nk_haversine_f64, a_lats, a_lons, b_lats, b_lons, out, count),
        .f32 => distanceImpl(types.F32, c.nk_haversine_f32, a_lats, a_lons, b_lats, b_lons, out, count),
        else => @compileError("unsupported dtype for haversine"),
    };
}

pub fn vincenty(comptime dtype: types.DType, a_lats: []const u8, a_lons: []const u8, b_lats: []const u8, b_lons: []const u8, out: []u8, count: usize) Error!void {
    return switch (dtype) {
        .f64 => distanceImpl(types.F64, c.nk_vincenty_f64, a_lats, a_lons, b_lats, b_lons, out, count),
        .f32 => distanceImpl(types.F32, c.nk_vincenty_f32, a_lats, a_lons, b_lats, b_lons, out, count),
        else => @compileError("unsupported dtype for vincenty"),
    };
}

test "geospatial comptime dtype API writes f64 output" {
    const lats = [_]types.F64{ 0, 0.1 };
    const lons = [_]types.F64{ 0, 0.2 };
    var out = [_]types.F64{ 1, 1 };
    const lats_bytes = std.mem.sliceAsBytes(lats[0..]);
    const lons_bytes = std.mem.sliceAsBytes(lons[0..]);
    const out_bytes = std.mem.sliceAsBytes(out[0..]);

    try haversine(.f64, lats_bytes, lons_bytes, lats_bytes, lons_bytes, out_bytes, lats.len);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), out[0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), out[1], 1e-9);
    try vincenty(.f64, lats_bytes, lons_bytes, lats_bytes, lons_bytes, out_bytes, lats.len);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), out[0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), out[1], 1e-9);
}

test "geospatial comptime dtype API writes f32 output" {
    const lats = [_]types.F32{ 0, 0.1 };
    const lons = [_]types.F32{ 0, 0.2 };
    var out = [_]types.F32{ 1, 1 };
    const lats_bytes = std.mem.sliceAsBytes(lats[0..]);
    const lons_bytes = std.mem.sliceAsBytes(lons[0..]);
    const out_bytes = std.mem.sliceAsBytes(out[0..]);

    try haversine(.f32, lats_bytes, lons_bytes, lats_bytes, lons_bytes, out_bytes, lats.len);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), out[1], 1e-4);
    try vincenty(.f32, lats_bytes, lons_bytes, lats_bytes, lons_bytes, out_bytes, lats.len);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), out[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(types.F32, 0), out[1], 1e-4);
}

test "geospatial comptime dtype API reports invalid inputs" {
    const coords = [_]types.F32{0};
    var out = [_]types.F32{0};
    const coords_bytes = std.mem.sliceAsBytes(coords[0..]);
    const out_bytes = std.mem.sliceAsBytes(out[0..]);
    try std.testing.expectError(Error.EmptyInput, haversine(.f32, coords_bytes, coords_bytes, coords_bytes, coords_bytes, out_bytes, 0));
    try std.testing.expectError(Error.InputTooSmall, vincenty(.f32, coords_bytes[0..3], coords_bytes, coords_bytes, coords_bytes, out_bytes, 1));
    try std.testing.expectError(Error.OutputTooSmall, vincenty(.f32, coords_bytes, coords_bytes, coords_bytes, coords_bytes, out_bytes[0..3], 1));
}
