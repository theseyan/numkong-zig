const std = @import("std");
const c = @import("c.zig").raw;
const types = @import("types.zig");

pub const Error = error{UnsupportedDType};

pub fn packedSize(comptime dtype: types.DType, vector_count: usize, depth: usize) Error!usize {
    return switch (dtype) {
        .bf16 => c.nk_maxsim_packed_size_bf16(vector_count, depth),
        .f32 => c.nk_maxsim_packed_size_f32(vector_count, depth),
        .f16 => c.nk_maxsim_packed_size_f16(vector_count, depth),
        else => Error.UnsupportedDType,
    };
}

pub fn pack(comptime dtype: types.DType, vectors: []const u8, vector_count: usize, depth: usize, stride_bytes: usize, out: []u8) Error!void {
    std.debug.assert(out.len >= try packedSize(dtype, vector_count, depth));
    switch (dtype) {
        .bf16 => c.nk_maxsim_pack_bf16(@ptrCast(@alignCast(vectors.ptr)), vector_count, depth, stride_bytes, out.ptr),
        .f32 => c.nk_maxsim_pack_f32(@ptrCast(@alignCast(vectors.ptr)), vector_count, depth, stride_bytes, out.ptr),
        .f16 => c.nk_maxsim_pack_f16(@ptrCast(@alignCast(vectors.ptr)), vector_count, depth, stride_bytes, out.ptr),
        else => return Error.UnsupportedDType,
    }
}

pub fn compute(comptime dtype: types.DType, query_packed: []const u8, document_packed: []const u8, query_count: usize, document_count: usize, depth: usize) Error!types.F64 {
    return switch (dtype) {
        .bf16 => blk: {
            var result: types.F32 = undefined;
            c.nk_maxsim_packed_bf16(query_packed.ptr, document_packed.ptr, query_count, document_count, depth, &result);
            break :blk result;
        },
        .f32 => blk: {
            var result: types.F64 = undefined;
            c.nk_maxsim_packed_f32(query_packed.ptr, document_packed.ptr, query_count, document_count, depth, &result);
            break :blk result;
        },
        .f16 => blk: {
            var result: types.F32 = undefined;
            c.nk_maxsim_packed_f16(query_packed.ptr, document_packed.ptr, query_count, document_count, depth, &result);
            break :blk result;
        },
        else => Error.UnsupportedDType,
    };
}

test "maxsim f32 pack and compute" {
    const vector = [_]types.F32{ 1, 2 };
    var query_packed: [4096]u8 = undefined;
    var document_packed: [4096]u8 = undefined;
    const size = try packedSize(.f32, 1, 2);
    try pack(.f32, std.mem.sliceAsBytes(vector[0..]), 1, 2, 2 * @sizeOf(types.F32), query_packed[0..size]);
    try pack(.f32, std.mem.sliceAsBytes(vector[0..]), 1, 2, 2 * @sizeOf(types.F32), document_packed[0..size]);
    try std.testing.expectApproxEqAbs(@as(types.F64, 0), try compute(.f32, query_packed[0..size], document_packed[0..size], 1, 1, 2), 1e-6);
}

test "maxsim packed size supports all dtypes" {
    try std.testing.expect((try packedSize(.bf16, 1, 2)) > 0);
    try std.testing.expect((try packedSize(.f32, 1, 2)) > 0);
    try std.testing.expect((try packedSize(.f16, 1, 2)) > 0);
}
