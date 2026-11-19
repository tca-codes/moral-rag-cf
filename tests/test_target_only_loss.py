import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_target_only_loss_matches_standard_loss():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from mvrag.agg_train import collate, target_only_loss

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128))
    enc = []
    for plen, tlen in [(20, 4), (8, 7), (15, 1)]:
        ids = torch.randint(3, 97, (plen + tlen,)).tolist()
        enc.append((ids, [-100] * plen + ids[plen:]))
    i_r, m_r, l_r = collate(enc, 0)
    i_l, m_l, l_l = collate(enc, 0, left=True)
    ref = model(input_ids=i_r, attention_mask=m_r, labels=l_r).loss
    new = target_only_loss(model, i_l, m_l, l_l)
    assert abs(ref.item() - new.item()) < 1e-5
    assert (l_l[:, -1] != -100).all() and (m_l[:, -1] == 1).all()  # targets sit at the end
