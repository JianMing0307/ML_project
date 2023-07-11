import numpy as np
import pandas as pd
import streamlit as st

st.title("測試首頁")
st.write("HaHaHa")

a = 100
st.write(a)
st.write("------------------------------------\n")
st.write("表格---------------------------------")
df = pd.DataFrame({"F1":[1,2,3,4], "F2":[11,22,33,44]})
st.write(df)

st.write("------------------------------------\n")
st.write("核取方塊")
cb = st.checkbox("是否外送?")
if cb:
    st.info("外送")

st.write("------------------------------------\n")
st.write("選項按鈕")
gender = st.radio("性別:",('M','F','None'))
st.success(gender)

st.write("------------------------------------\n")
st.write("下拉選單")
option = st.selectbox("choose the favorite food",['pizza','hamburger','fried rice'])
st.success(option)
"你的答案:",option

st.write("------------------------------------\n")
st.write("進度條")
import time
aa = st.empty()
bar = st.progress(0)
for i in range(100):
    aa.text(f"current progress:{i+1}%")
    bar.progress(i+1)
    time.sleep(0.05)

st.write("------------------------------------\n")
st.write("按鈕")
btn = st.button("確定")
if btn:
    st.info("已確認")

st.write("------------------------------------\n")
st.write("滑桿")
num = st.slider("請選擇數量:",1,5)
"num=",num

st.write("------------------------------------\n")
st.write("檔案上傳")
loader = st.file_uploader("請選擇csv檔:")

if loader is not None:   
    df2 = pd.read_csv(loader,header=None)
    st.dataframe(df2)
    st.table(df2.iloc[:2])

st.write("------------------------------------\n")
st.write("隱藏欄位")
hidden = st.expander("按下之後展開")
hidden.write("DFGHJJKL")

st.write("------------------------------------\n")
st.write("圖片上傳+圖片展示")
img = st.file_uploader("請選擇圖檔:",type=['png','jpg','jpeg'])
if img is not None:
    st.image(img)

st.write("------------------------------------\n")
st.write("側邊欄")
side01 = st.sidebar.button("Click Me")
side02 = st.sidebar.checkbox("Ok?")

st.write("------------------------------------\n")
st.write("分欄")
left, right = st.columns(2)

with left:
    st.radio("select one:",[1,2])
#left.write("btn3")
with right:
    btn0 = st.button("確定")
#right.write("BBBBBB")
if btn0:
    st.info("以確認")
