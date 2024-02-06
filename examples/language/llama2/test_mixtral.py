import torch
from pretrain import replace_moe_layer
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM


def test_mixtral_replace():
    torch.backends.cuda.matmul.allow_tf32 = False
    config = MixtralConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=8,
        use_cache=False,
    )

    mixtral_model = MixtralForCausalLM(config)
    cai_model = MixtralForCausalLM(config)
    replace_moe_layer(cai_model, enable_kernel=True)
    mixtral_model = mixtral_model.model.layers[0].block_sparse_moe.cuda()
    cai_model = cai_model.model.layers[0].block_sparse_moe.cuda()

    cai_model.gate_weight.data = mixtral_model.gate.weight.data.clone().detach()
    for i in range(8):
        cai_model.experts.wi_gate.data[i] = mixtral_model.experts[i].w1.weight.data.T.clone().detach()
        cai_model.experts.wi_up.data[i] = mixtral_model.experts[i].w3.weight.data.T.clone().detach()
        cai_model.experts.wo.data[i] = mixtral_model.experts[i].w2.weight.data.T.clone().detach()

    inputs = torch.randn(2, 1, 256).cuda()
    mixtral_output = mixtral_model(inputs)[0]
    cai_output = cai_model(inputs)[0]
    assert torch.allclose(mixtral_output, cai_output), f"diff {mixtral_output - cai_output}\n"


if __name__ == "__main__":
    test_mixtral_replace()
