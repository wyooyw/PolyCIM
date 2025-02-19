1.CIMSet用来设置Macro的哪些列关闭。目前为了简单，先设置成了全0，所有列都开启。后续可以按需求开启对应列
2.CIMOutput目前直接放在了CIMComputeDense后面。后续如果n_row > 1，则这里还需要灵活处理一下
3.当最内两层大小小于Macro大小时，可能还需要padding一下，不然会有bug。可通过调大n_comp或n_bcol来触发？
4.output可能会错位相加。任何一个domain axis都会改变output，但整个domain到output不是单射。这个改起来会很复杂，之后再说吧。
