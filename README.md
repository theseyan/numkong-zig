# numkong-zig

Idiomatic Zig bindings for [NumKong](https://github.com/ashvardanian/NumKong), a SIMD-accelerated collection of computational kernels for vectors, sets, sparse data, geometry, and batched similarity search.

Built for Zig `0.16.0` and NumKong `7.6.0`.

## Installation

Fetch the package:

```bash
zig fetch --save=numkong_zig https://github.com/theseyan/numkong-zig/archive/refs/tags/{VERSION}.tar.gz
```

Then import the module from your `build.zig`:

```zig
const numkong_dep = b.dependency("numkong_zig", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("numkong", numkong_dep.module("numkong"));
```

## Usage

Most kernel APIs take a compile-time `numkong.DType` and byte slices. The dtype selects the NumKong kernel at compile time, while byte slices keep the API thin and work for packed/sub-byte formats.

```zig
const std = @import("std");
const numkong = @import("numkong");

pub fn main() !void {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 5, 6 };

    const product = try numkong.dot.dot(
        .f32,
        std.mem.sliceAsBytes(a[0..]),
        std.mem.sliceAsBytes(b[0..]),
        a.len,
    );

    std.debug.print("dot = {d}\n", .{product});
}
```

Element-wise kernels write into caller-owned output buffers:

```zig
const std = @import("std");
const numkong = @import("numkong");

pub fn scale(xs: []const f32, out: []f32) !void {
    try numkong.each.scale(
        .f32,
        std.mem.sliceAsBytes(xs),
        2.0,
        1.0,
        std.mem.sliceAsBytes(out),
    );
}
```

Packed and sub-byte dtypes use logical counts or dimensions, not just byte lengths. For example, `.u1` vectors use `numkong.U1x8` storage but counts are still expressed in bits.

Scalar casts use the same dtype style:

```zig
const half = numkong.cast.fromF32(.f16, 1.5);
const value = numkong.cast.toF32(.f16, half);
```

## Modules

| Module | Purpose |
|---|---|
| `types` | NumKong metadata, dtypes, capability detection, and dispatch lookup. |
| `scalar` | Scalar math helpers, saturating integer operations, and low-precision ordering helpers. |
| `cast` | `fromF32`/`toF32` scalar casts and slice conversions between supported NumKong dtypes. |
| `dot` | Dense dot products and complex conjugating dot products. |
| `spatial` | Single-vector Euclidean, squared Euclidean, and angular distances. |
| `probability` | Kullback-Leibler and Jensen-Shannon divergences. |
| `each` | Element-wise scale, sum, blend, FMA, sine, cosine, and atan kernels. |
| `reduce` | Moments and min/max reductions. |
| `set` | Hamming and Jaccard kernels for binary and integer sets. |
| `sparse` | Sorted sparse intersections and sparse weighted dot products. |
| `geospatial` | Haversine and Vincenty distance batches. |
| `curved` | Bilinear and Mahalanobis forms. |
| `mesh` | RMSD, Kabsch, and Umeyama mesh alignment kernels. |
| `dots` | Packed and symmetric batched dot-product kernels. |
| `sets` | Packed and symmetric batched Hamming/Jaccard kernels. |
| `spatials` | Packed and symmetric batched spatial-distance kernels. |
| `maxsim` | Packed MaxSim kernels for query/document vector groups. |

Core type aliases such as `numkong.F32`, `numkong.F16`, `numkong.BF16`, `numkong.F32C`, and `numkong.DType` are re-exported at the root.

## Development

Run tests:

```bash
zig build test
```

## Benchmarks

The benchmark suite uses [zBench](https://github.com/hendriknielaender/zBench) and covers representative bindings across the library.

```bash
zig build bench -Doptimize=ReleaseFast
```

## License

MIT License. See [LICENSE](LICENSE).
