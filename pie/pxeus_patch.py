#!/usr/bin/env python3
"""Make PhoneticXeus's int8 ONNX export (allan-valin/phoneticxeus-onnx, built for onnxruntime-web) run on the CPU
runtime, which has no int8 ConvInteger: each DynamicQuantizeLinear → ConvInteger → Cast → Mul(scales) chain becomes
one float Conv with the dequantized weights ((W_q - zero point) × scale). Everything else stays int8.

    python pie/pxeus_patch.py      # writes $CHIMERA_PIE_DATA/hf/pxeus-convfloat.onnx
"""
import collections, glob, os
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

DATA = Path(os.environ.get("CHIMERA_PIE_DATA", Path.home() / "AI/chimera-pie-data"))
src = glob.glob(str(DATA / "hf/models--allan-valin--phoneticxeus-onnx/snapshots/*/model.int8.onnx"))[0]
m = onnx.load(src)
g = m.graph
init = {i.name: i for i in g.initializer}
prod = {o: n for n in g.node for o in n.output}
cons = collections.defaultdict(list)
for n in g.node:
    for i in n.input:
        cons[i].append(n)


def val(name):
    if name in init:
        return numpy_helper.to_array(init[name])
    return numpy_helper.to_array(prod[name].attribute[0].t)


repl, drop, new_inits = {}, set(), []
for n in [n for n in g.node if n.op_type == "ConvInteger"]:
    xq, wq, _xzp, wzp = n.input
    dql = prod[xq]
    cast = cons[n.output[0]][0]
    mul = cons[cast.output[0]][0]
    scales = prod[[i for i in mul.input if i != cast.output[0]][0]]
    w_scale = [i for i in scales.input if i != dql.output[1]][0]
    wq_a = val(wq)
    shape = (-1,) + (1,) * (wq_a.ndim - 1)
    w = (wq_a.astype(np.float32) - val(wzp).astype(np.float32).reshape(shape)) * val(w_scale).astype(np.float32).reshape(shape)
    wname = wq.replace("_quantized", "_float")
    new_inits.append(numpy_helper.from_array(w, wname))
    conv = helper.make_node("Conv", [dql.input[0], wname], [mul.output[0]], name=n.name.replace("_quant", "_float"))
    conv.attribute.extend(n.attribute)
    repl[id(n)] = conv
    drop.update([id(cast), id(mul)])
nodes = [repl.get(id(n), n) for n in g.node if id(n) not in drop]
while True:  # dead code: quantizers and scale products that fed only the removed nodes
    live = {o.name for o in g.output} | {i for n in nodes for i in n.input}
    keep = [n for n in nodes if any(o in live for o in n.output)]
    if len(keep) == len(nodes):
        break
    nodes = keep
del g.node[:]
g.node.extend(nodes)
g.initializer.extend(new_inits)
used = {i for n in nodes for i in n.input}
keep_init = [i for i in g.initializer if i.name in used]
del g.initializer[:]
g.initializer.extend(keep_init)
out = DATA / "hf/pxeus-convfloat.onnx"
onnx.save(m, str(out))
print("wrote", out)
