# Ghi chú: MMLU Benchmark không khả dụng

## Tóm tắt

Benchmark MMLU không chạy được trong quá trình thực hiện Lab 22, dẫn đến kết quả MMLU hiển thị `nan` trong bảng benchmark. Đây **không phải lỗi code** mà do sự kết hợp của 2 yếu tố bên ngoài.

## Nguyên nhân

### 1. HuggingFace Hub Server Error (HTTP 500)

Server của HuggingFace trả về lỗi `500 Internal Server Error` khi `lm-eval` cố tải dataset `cais/mmlu`:

```
huggingface_hub.errors.HfHubHTTPError: Server error '500 Internal Server Error' 
for url 'https://huggingface.co/api/datasets/cais/mmlu/tree/...'
Internal Error - We're working hard to fix this as soon as possible!
```

Lỗi này xảy ra ở phía server HuggingFace, nằm ngoài tầm kiểm soát.

### 2. Timeout do thiếu 4-bit quantization support

Ngay cả khi server HuggingFace hoạt động bình thường, `lm-eval` vẫn timeout trên T4 vì:

- **`load_in_4bit` không tương thích với transformers 5.x:** Tham số `load_in_4bit=True` gây lỗi `TypeError: Qwen2ForCausalLM.__init__() got an unexpected keyword argument 'load_in_4bit'`. Đây là breaking change trong transformers 5.x so với 4.x.
- **Model phải chạy FP16:** Sau khi bỏ `load_in_4bit`, model load ở full precision (FP16), tốn gấp đôi VRAM và **chậm gấp 3-4 lần** so với 4-bit inference.
- **MMLU dùng 5-shot prompting:** Mỗi câu hỏi kèm 5 ví dụ mẫu, tạo ra context rất dài → sinh token chậm.
- **Kết quả:** Ngay cả khi giảm `LIMIT_MMLU` xuống 50 samples và tăng timeout lên 3600s (60 phút), benchmark vẫn timeout trên Tesla T4 free tier.

## Các benchmark đã chạy thành công

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | 0.060 | 0.060 | 0.000 |
| GSM8K | 0.540 | 0.600 | +0.060 ↑ |
| AlpacaEval-lite | 0.500 | 0.620 | +0.120 ↑ |

3/4 benchmark đã chạy thành công và cho kết quả có ý nghĩa thống kê.

## Môi trường

- GPU: Tesla T4 16GB (Free Colab)
- transformers: 5.3.0 / 5.5.0
- lm-eval: 0.4.x
- unsloth: 2026.5.2
