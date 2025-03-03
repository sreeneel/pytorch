import sys
import os
import torch

if len(sys.argv) != 2:
    print('Usage: 5d_upsample_ex <device: cpu|mps|cuda>')
    exit(1)

device=sys.argv[1]
print(f'Running on device: {device}')

def runTest(x_shape, sz):
    print(f'Process pid=: {os.getpid()}')
    print(f'Generate a system-trace as follows: "xctrace record --template \'Metal System Trace\' --attach {os.getpid()}"')
    input(f'Press any key to start the test...')
    torch.mps.profiler.start(mode="interval,event", wait_until_completed=True)
    (N, C, D, H, W) = x_shape
    num_el = N * C * D * H * W
    x=torch.arange(1, num_el+1, dtype=torch.float32, device=device).view(N, C, D, H, W)
    # x=torch.arange(1, num_el+1, dtype=torch.float32, device=device).view(N, C, D, H, W)
    x.requires_grad = True
    upsampleFn = torch.nn.Upsample(size=sz, mode='nearest')
    output = upsampleFn(x)
    output.backward(torch.ones_like(output))
    torch.mps.profiler.stop()

    torch.set_printoptions(profile="full")
    print(f'x={x}: {x.shape}\n')
    print(f'"torch.nn.Upsample(size={sz}), mode="nearest")"')
    print(f'{output}: {output.shape}\n')
    print(f'x.grad={x.grad}, sh={x.grad.shape}')

    # Compare result with the default CPU implementation
    x_cpu=torch.arange(1, num_el+1, dtype=torch.float32, device="cpu").view(N, C, D, H, W)
    x_cpu.requires_grad = True
    output_cpu = upsampleFn(x_cpu)
    output_cpu.backward(torch.ones_like(output_cpu))
    print(f"x == x_cpu: {torch.equal(x.cpu(), x_cpu)}")

# runTest((1,1,1,2,3), (1,2,4)) #test-00
# runTest((1,1,1,2,3), (1,4,4)) #test-03
# runTest((1,1,2,2,3), (2,4,4)) #test-04 depth > 1, 
# runTest((1,1,1,2,3), (2,3,4)) #test-05
# runTest((1,2,2,2,3), (3,3,4)) #test-06 num-chan > 1; //<--req. %gradInDepth
# runTest((1,1,1,2,3), (10, 8, 8)) #test-07
# runTest((1,1,1,2,3), (12, 10, 10)) #test-08
# runTest((1,1,2,2,3), (12, 10, 10)) #test-09
# runTest((1,1,2,2,3), (10, 10, 10)) #test-10
# runTest((1,1,2,2,3), (10, 11, 12)) #test-11
# runTest((1,1,1,2,3), (33, 13, 12)) #test-12
# runTest((2,1,1,2,3), (33, 13, 12)) #test-13 batchsz > 1
# runTest((2,2,1,2,3), (33, 13, 12)) #test-14 batchsz > 1
runTest((2,3,1,2,3), (33, 11, 12)) #test-14 

# runTest((2,3,1,2,3), (33, 11, 12)) #test-14
# runTest((2,2,1,2,3), (1,2,4)) #test-15
# runTest((2,2,2,3,4), (3,4,5)) #test-15

# runTest((1,1,1,2,3), (1,3,2)) #test-01 : Downsampling on W (3-->2)
# runTest((1,1,1,2,3), (1,4,2)) #test-02 : Downsampling on W (3-->2)

# runTest((1,1,1,3,2), (1,2,3)) #test-02 : Downsampling on H (3-->2)
# runTest((1,1,1,3,2), (1,2,2)) #test-02 : Downsampling on H (3-->2)

# runTest((1,1,3,2,2), (1,2,3)) #test-02 : Downsampling on D (3-->2)
# runTest((1,1,3,2,2), (1,2,2)) #test-02 : Downsampling on D (3-->2)
