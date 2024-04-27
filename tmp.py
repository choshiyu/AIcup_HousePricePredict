# 距離1 3 5
model_mlp - MAPE: [42.91546]
model_RandomForest - MAPE: 12.050616495084475

# 把不在台灣的train資料刪掉
# 距離調近0.7 2 4 
model_mlp - MAPE: [20.738726]
model_RandomForest - MAPE: 19.153267025886922  

# 單純距離調近0.7 2 4 
model_mlp - MAPE: [11.839245]
model_RandomForest - MAPE: 12.081936645899873

# 加了焚化廠跟垃圾掩埋
model_mlp - MAPE: [44.568695]
model_RandomForest - MAPE: 12.20403915126401

# 0.7 2 4  rf feature select 25特徵 焚化廠跟垃圾掩埋
model_mlp - MAPE: [43.129555]
model_RandomForest - MAPE: 12.173922018487646

# 不要rf feature select
model_RandomForest - MAPE: 12.076491646996049

# 附近設施改成積分制+1 -1
# 地區分開不同模型跑