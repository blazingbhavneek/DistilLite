import GPUtil
import psutil


def available_memory():
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    available_ram = round(ram.available / 1e9, 2)
    available_disk = round(disk.free / 1e9, 2)
    available_gpus = []
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        available_gpus.append(
            {
                "name": gpu.name,
                "memoryFree": gpu.memoryFree,
            }
        )

    return {
        "available_ram": available_ram,
        "available_disk": available_disk,
        "available_gpus": available_gpus,
    }


if __name__ == "__main__":
    memory_info = available_memory()
    print(f"Available RAM: {memory_info['available_ram']} GB")
    print(f"Available Disk: {memory_info['available_disk']} GB")
    print("Available GPUs:")
    for gpu in memory_info["available_gpus"]:
        print(f"  - {gpu['name']}: {gpu['memoryFree']} MB free")
    print("Hardware information retrieved successfully.")
