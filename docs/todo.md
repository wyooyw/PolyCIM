1.CIMSet用来设置Macro的哪些列关闭。目前为了简单，先设置成了全0，所有列都开启。后续可以按需求开启对应列
2.CIMOutput目前直接放在了CIMComputeDense后面。后续如果n_row > 1，则这里还需要灵活处理一下
