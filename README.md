## 项目背景：
  地址元素识别可以抽取地址中不同的地址元素，同时也可以作为其它项目任务的基础。

## 使用:
  train_eval.py：训练与评估模型（可以选择IDCNN膨胀卷积网络或者BILSTM）
  address_segment_service：使用Tornado部署模型（可以进行多线程部署），从而通过http协议访问服务
## 示例：
 在浏览器地址栏输入：http://localhost:5002/fact?inputStr=江苏省南京市六合区雄州街道雄州南路333号冠城大通南郡25幢1单元502室
 {'string': '江苏省南京市六合区雄州街道雄州南路333号冠城大通南郡25幢1单元502室', 
 'entities': [{'word': '江苏省', 'start': 0, 'end': 3, 'type': 'XZQHS'}, 
 {'word': '南京市', 'start': 3, 'end': 6, 'type': 'XZQHCS'}, 
 {'word': '六合区', 'start': 6, 'end': 9, 'type': 'XZQHQX'}, 
 {'word': '雄州街道', 'start': 9, 'end': 13, 'type': 'JD1'}, 
 {'word': '雄州南路', 'start': 13, 'end': 17, 'type': 'JD2'}, 
 {'word': '333号', 'start': 17, 'end': 21, 'type': 'MP1'}, 
 {'word': '冠城大通南郡', 'start': 21, 'end': 27, 'type': 'MP2'}, 
 {'word': '25幢', 'start': 27, 'end': 30, 'type': 'MP3'}, 
 {'word': '1单元', 'start': 30, 'end': 33, 'type': 'DYS1'}, 
 {'word': '502室', 'start': 33, 'end': 37, 'type': 'DYS2'}]}