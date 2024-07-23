import torch
from model import dsc
from torchsummary import summary
import torch.profiler as profiler
from thop import profile
from ptflops import get_model_complexity_info
from torch.profiler import profile, record_function, ProfilerActivity
if __name__ == "__main__":
    model = dsc()

    device = torch.device("cuda")
    random_input = torch.randn(1, 1, 4200).to(device)
    model = model
    model.to(device)
    summary(model, input_size=(1, 4200), batch_size=1)

    flops, params = get_model_complexity_info(model, (1, 4200), as_strings=True, print_per_layer_stat=True)

    # 使用TorchScript Profiler进行性能分析
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            output = model(random_input)

    # 打印分析结果
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    input_data = torch.randn(1, 1, 4200).cuda()
    # 获取GPU内存使用前的状态
    initial_memory_stats = torch.cuda.memory_stats()
    # 运行模型
    output = model(input_data)
    # 获取GPU内存使用后的状态
    final_memory_stats = torch.cuda.memory_stats()

    # 计算内存访问量
    memory_access = final_memory_stats["allocated_bytes.all.current"] - initial_memory_stats[
        "allocated_bytes.all.current"]
    print("Estimated memory access: {:.2f} Kbytes".format(memory_access/1024))

    # Analyze the inference time
    iterations = 50  # Repeated rounds of calculation

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU warm-up
    for _ in range(50):
        _ = model(random_input)

    # Speed measurement
    times = torch.zeros(iterations)  # Store the time of each iteration
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(random_input)
            ender.record()
            # Synchronized GPU time
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # Calculation time
            times[iter] = curr_time
            # print(curr_time)

    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

