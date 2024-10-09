import gradio as gr
import time
def pdf_extract(file_path, sentences_1):
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # from FlagEmbedding import FlagModel
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel('BAAI/bge-m3',
                           use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
    import pdfplumber
    similarity_list = []
    texts_list = []
    embeddings_1 = model.encode(sentences_1,
                                batch_size=12,
                                max_length=8192,
                                # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                )['dense_vecs']
    with pdfplumber.open(file_path) as pdf:
        pages = pdf.pages
        for page in pages:  # 指定页码
            text = page.extract_text()  # 提取文本
            texts_list.append(text)
    for sentences_2 in texts_list:
        embeddings_2 = model.encode(sentences_2)['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        similarity_list.append(similarity)
    index = similarity_list.index(max(similarity_list))
    #保存与问题相关的文档所在页
    with open('test_pdf_extract_sim.txt', 'w', encoding='utf-8') as f:
        f.write(texts_list[index])
    f.close()
    return texts_list[index]
def stram_greet(sentences, file, is_file):
    # import time
    import random
    streaming_output = ''
    file_path = file
    from zhipuai import ZhipuAI
    pdf_path = file_path
    # t1 = time.perf_counter()
    #是否加入检索文档
    if is_file == '是':
        key_pages = pdf_extract(sentences_1=sentences, file_path=pdf_path)
        message=[
            {"role": "user",
             "content": "Here is some knowledge . Please read it and answer my questions."+key_pages
             },
            {"role": "user",
             "content": sentences
             },
        ]
        t2 = time.perf_counter()
    elif is_file == '否':
        message=[
            {"role": "user",
             "content": sentences
             },
        ]
    else:
        raise ValueError("针对是否输入PDF，请输入：“是”或者“否”!")
    client = ZhipuAI(api_key="b04ac68046990a131a822e469a1b195e.dMxNeQu5UXRTnMkZ")  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4",  # 请填写您要调用的模型名称
        messages=message,
        stream=True,
    )
    for chunk in response:
        streaming_output = streaming_output+chunk.choices[0].delta.content
        time.sleep(0.01)
        yield streaming_output
    t4 = time.perf_counter()
def UI_design():
    demo = gr.Interface(
        fn=stram_greet,
        inputs=[gr.Textbox(label='问题', lines=3,placeholder='请输入问题：'),
                gr.File(file_count='single',file_types=['file'], label='文件'),
                gr.Textbox(label='检索', lines=3,placeholder='是否需要检索文档，请输入是或者否')],
        outputs=gr.Textbox(label='输出', lines=3)
    )
    demo.launch(server_name='127.0.0.1',server_port=7862, share=False, show_error=False)
if __name__ == '__main__':
    sentences = 'Whats is the advantage of ChatGLM?'
    file_path = 'C:\\Users\\FXH\\Desktop\\research_articles'
    is_file = False
    UI_design()