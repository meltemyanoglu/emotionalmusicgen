import torch
import numpy as np
import soundfile as sf
from utils._modeltraining import EmotionalVAE
from utils import _latentspace

def print_latent_stats(latent_tokens, label="EmotionalVAE Latents"):
    """
    latent_tokens: numpy array, shape: [num_samples, quantizer_count, latent_dim]
    Katman bazında ortalama ve standart sapma değerlerini hesaplar ve yazdırır.
    """
    num_samples, quantizer_count, latent_dim = latent_tokens.shape
    means = latent_tokens.mean(axis=(0,2))  # her quantizer için ortalama
    stds = latent_tokens.std(axis=(0,2))    # her quantizer için std
    
    print(f"== {label} ==")
    print("Quantizer Katmanları İçin Ortalama Değerler:")
    for i in range(quantizer_count):
        print(f"  Katman {i}: {means[i]:.4f}")
    print("Quantizer Katmanları İçin Standart Sapmalar:")
    for i in range(quantizer_count):
        print(f"  Katman {i}: {stds[i]:.4f}")
    print("----------------------------------------------------")

def get_encodec_scale_from_audio(audio_path, lat_gen, device):
    """
    Verilen bir ses dosyasını EnCodec encoder'a vererek gerçek scale değerlerini elde eder.
    Not: Bu kod, EnCodec modelinizin encode() metodunun (codes, scale) çıktısı döndürdüğünü varsayar.
    """
    # Ses dosyasını yükle (mono olması bekleniyor)
    audio, sr = sf.read(audio_path)
    print(f"Ses dosyası yüklendi, örnekleme hızı: {sr}, Ses verisi shape: {np.array(audio).shape}")
    # Tensora çevir; batch boyutunu ekle
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoded = lat_gen.encodec_model.encode(audio_tensor)
        print("Encode metodunun çıktısı:")
        print(encoded)
        # Eğer encoded tuple ise scale = encoded[1]
        # Aksi halde scale değeri olmayabilir.
        try:
            scale = encoded[1]
        except IndexError:
            raise ValueError("Encode metodunun çıktısı scale bilgisini içermiyor!")
    return scale.cpu().numpy()

if __name__ == "__main__":
    # EmotionalVAE modelini yükle ve latent token üretimi yap
    latent_dim = 750
    hidden_dims = [512, 256, 128]
    condition_dim = 2
    model = EmotionalVAE(latent_dim, hidden_dims, condition_dim)
    model_path = 'vae_final.pt'
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("EmotionalVAE model yüklendi.")
    
    # Örnek duygusal koşul (valence, arousal)
    num_samples = 5
    condition = torch.tensor([0.7, 0.6])
    if len(condition.shape) == 1:
        condition = condition.unsqueeze(0)
    condition = condition.repeat(num_samples, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Latent token üretimi
    latent_tokens = model.sample(num_samples, condition, device)  # [num_samples, quantizer_count, latent_dim]
    latent_tokens_np = latent_tokens.detach().cpu().numpy()
    print_latent_stats(latent_tokens_np, label="EmotionalVAE Üretilen Latents")
    
    # EnCodec modelini yükleyelim
    lat_gen = _latentspace.LatentRepresentationGenerator(
        encodec_bandwidth=6.0,
        device=device,
        chunk_sizes=[5]
    )
    lat_gen._load_encodec_model()
    print("EnCodec decoder yüklendi.")
    
    # Gerçek ses dosyası üzerinden EnCodec encoder scale değerini alalım
    # Lütfen buraya test etmek için kullanılacak gerçek ses dosyasının yolunu yazınız.
    example_audio_path = "8.mp3"
    try:
        real_scale = get_encodec_scale_from_audio(example_audio_path, lat_gen, device)
        print("EnCodec encoder'dan elde edilen scale değeri:")
        print(real_scale)
        print("----------------------------------------------------")
    except Exception as e:
        print("EnCodec encoder scale değeri alınamadı, hata:", e)