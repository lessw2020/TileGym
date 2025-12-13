# Test bitwise abs implementation
import cuda.tile as ct
import torch

@ct.kernel
def test_abs_kernel(input, output_max, output_bitwise, TILE_SIZE: ct.Constant[int]):
    bid = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=torch.int32)
    
    x = ct.gather(input, (bid, offsets), check_bounds=True)
    x = ct.astype(x, torch.float32)
    
    # Method 1: max(x, -x)
    abs_max = ct.maximum(x, ct.negative(x))
    
    # Method 2: bitwise (clear sign bit)
    x_int = ct.bitcast(x, ct.int32)
    x_abs_int = ct.bitwise_and(x_int, 0x7FFFFFFF)
    abs_bitwise = ct.bitcast(x_abs_int, ct.float32)
    
    ct.scatter(output_max, (bid, offsets), abs_max, check_bounds=True)
    ct.scatter(output_bitwise, (bid, offsets), abs_bitwise, check_bounds=True)


def test():
    device = "cuda"
    batch, hidden = 4, 32
    x = torch.randn(batch, hidden, dtype=torch.float32, device=device)
    x[0, :4] = torch.tensor([-1.5, 2.3, -0.001, 0.0])  # Test specific values

    out_max = torch.empty_like(x)
    out_bitwise = torch.empty_like(x)

    ct.launch(torch.cuda.current_stream(), (batch,), test_abs_kernel, (x, out_max, out_bitwise, 32))
    torch.cuda.synchronize()

    print("Input (first 8):", x[0, :8])
    print("abs via max:    ", out_max[0, :8])
    print("abs via bitwise:", out_bitwise[0, :8])
    print()
    print("Match:", torch.allclose(out_max, out_bitwise))
    print("Max diff:", (out_max - out_bitwise).abs().max().item())


if __name__ == "__main__":
    test()
