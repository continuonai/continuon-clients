import pytest
import torch
from continuonbrain.wavecore.config import WaveCoreConfig
from continuonbrain.wavecore.models.spectral_lm import SpectralLanguageModel
from continuonbrain.wavecore.utils.trainer import WaveCoreTrainer

def test_wavecore_config():
    config = WaveCoreConfig.fast_loop()
    assert config.d_model == 64
    assert config.loop_type == "fast"

def test_model_forward():
    config = WaveCoreConfig(d_model=32, seq_len=16)
    model = SpectralLanguageModel(config)
    x = torch.randint(0, config.vocab_size, (2, config.seq_len))
    y = model(x)
    assert y.shape == (2, config.seq_len, config.vocab_size)

def test_trainer_step():
    config = WaveCoreConfig(d_model=32, seq_len=16)
    trainer = WaveCoreTrainer(config)
    loss = trainer.train_step()
    assert isinstance(loss, float)
    assert loss > 0

@pytest.mark.slow
def test_sanity_check():
    config = WaveCoreConfig(d_model=32, seq_len=16)
    trainer = WaveCoreTrainer(config)
    res = trainer.run_sanity_check(steps=10)
    assert res["status"] == "ok"
    assert res["final_loss"] < 10.0
