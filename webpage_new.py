import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import skew, kurtosis
import numpy as np

with st.sidebar:
    st.image("https://media.licdn.com/dms/image/C5612AQHqsSICWgrZFQ/article-cover_image-shrink_600_2000/0/1615514764556?e=2147483647&v=beta&t=SXpi8uqhHzyP1Eu9J21RRSfbFrdNyRaBdwFAImhGqaM",width=150)
    st.title('Contents')
    sections=['Theoretical Background','Data Analysis','Data Preprocessing','Model Building','Conclusion']
    choice=st.selectbox("",sections)

def newpar(i=3):
    for j in range(0,i):
        st.write('')
if choice=="Theoretical Background":
    tab1,tab2,tab3,tab4=st.tabs(['pH','Turbidity','Hardness',"Solids"])
    with tab1:
        st.title('Important Elements in Data')
        st.write('')
        st.subheader('pH')
        st.write('수질의 산성도와 알칼리도를 나타냄')
        st.write('일반적으로 6.5~8.5가 적정 범위임')
        newpar()
        st.image('https://waterqualitysolutions.com.au/wp-content/uploads/2024/09/ph-scale.png')
        newpar()
    with tab2:
        st.subheader('Turbidity(탁도)')
        st.write('물의 투명도를 나타냄')
        st.write('높은 탁도는 빛 침투를 줄이며, 광합성 등에 악영향을 끼침')
        st.write('낮을수록 좋음')
        newpar(3)
        st.image("https://www.camlab.co.uk/media/wysiwyg/trubidity-768x362.png")
        newpar(5)
    with tab3:
        st.subheader("Hardness(경도)")
        st.write("물속의 칼슘 및 마그네슘 농도")
        st.write("60~120mg/L가 적당함")
        newpar()
        st.image("https://cdn1.byjus.com/wp-content/uploads/2020/11/Hardness-of-Water-Temporary-and-Permanent-Hardness-in-Water-.png")
        newpar(9)
    with tab4:
        st.subheader("Solids(소)")
        st.write("수질에 포함된 고형물의 농도")
        st.write("일반적으로 25mg/L 이하")
        newpar()
        st.image("https://t4.ftcdn.net/jpg/01/18/85/23/360_F_118852383_kjyToVFqvQ9T1rNlfrYQuYiAlmtqZTU9.jpg")
        newpar(5)
    st.markdown('---')
    st.subheader('Sources:')
    urls=["https://sierrastreamsinstitute.org/monitoring/water-quality-parameters/","https://dnr.mo.gov/water/hows-water/monitoring-data/quality-assessment/testing-parameters","https://www.oregon.gov/deq/wq/Pages/Hardness-Dependent-Metals-Aquatic-Life-Criteria.aspx","https://scienceon.kisti.re.kr/srch/selectPORSrchReport.do?cn=KOSEN0000000758137&utm_source=chatgpt.com"]
    refs=["Sierra Streams monitoring water quality","Missouri Department of Natural Resources","Oregon Government Website","Scienceon"]
    refurl={'url':urls,'refs':refs}
    for url,ref in zip(refurl['url'],refurl['refs']):
        st.write(f"{ref}: [{url}]({url})")
elif choice=="Data Analysis":
    st.title(choice)
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    data = pd.read_csv('water_survive.csv')
    mapping = {'river': 0, 'lake': 1, 'ocean': 2}
    data['Origin'] = data['Origin'].replace(mapping)

    # 결측값 제거
    data_cleaned = data.dropna()

    output_file_path = 'data_dropna.csv'
    data.to_csv(output_file_path, index=False)
    output_file_path

    # 상관관계 히트맵
    fig=plt.figure(figsize=(10, 8))
    correlation_matrix = data_cleaned.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    st.pyplot(fig)
    arr=np.array((abs(correlation_matrix.values)))
    i,j=arr.shape
    max=-9999999999
    x=0
    y=0
    for l in range(0,i):
        for m in range(0,j):
            if int(arr[l][m])==1:
                arr[l][m]=0
            if arr[l][m]>max and l!=m:
                if l==10 or m==10:
                    max=arr[l][m]
                    y=l
                    x=m
    st.write("fig1.Feature Correlation Heatmap")
    newpar(4)
    st.write(f"{(data.columns[y])} 가 survivability와 가장 관련이 높고 {np.sort(arr)[-1][-1]}의 값을 가진다")

elif choice=='Data Preprocessing':
    st.title(choice)
    tabs = st.tabs(["dropna", 
                "pH", 
                "Hardness", 
                "Solids",
                "ph_Sulfate_corr", 
                "Solids_Sulfate_corr"])

# Dropna 데이터
    with tabs[0]:
        st.subheader("According to dropna")
        df = pd.read_csv('fixed_dropna.csv')
        st.dataframe(data=df.head(10))
        newpar(2)
        skewsum = 0
        kurtsum = 0
        for idx in df.columns:
            if idx == "ph":
                skewsum += skew(2 * df[idx])
                kurtsum += kurtosis(df[idx])
        datlen = len(df.columns)
        st.write(f"Skewness: {skewsum / datlen}")
        st.write(f"Kurtosis: {kurtsum / datlen}")
        st.write("결측치를 모두 drop함")

    # pH 데이터
    with tabs[1]:
        st.subheader("According to pH")
        df = pd.read_csv('ph_group_improved.csv')
        st.dataframe(data=df.head(10))
        newpar(2)
        skewsum = 0
        kurtsum = 0
        for idx in df.columns:
            if idx == "ph":
                skewsum += skew(2 * df[idx])
                kurtsum += kurtosis(df[idx])
        datlen = len(df.columns)
        st.write(f"Skewness: {skewsum / datlen}")
        st.write(f"Kurtosis: {kurtsum / datlen}")
        st.write("pH를 반올림해 그룹화함")
        st.write("pH의 결측치는 drop함")
        st.write("각 그룹의 평균값을 가지고 결측치를 채움")
    # Hardness 데이터
    with tabs[2]:
        st.subheader("According to Hardness")
        df = pd.read_csv('hardness_group_improved.csv')
        st.dataframe(data=df.head(10))
        newpar(2)
        skewsum = 0
        kurtsum = 0
        for idx in df.columns:
            if idx == "ph":
                skewsum += skew(2 * df[idx])
                kurtsum += kurtosis(df[idx])
        datlen = len(df.columns)
        st.write(f"Skewness: {skewsum / datlen}")
        st.write(f"Kurtosis: {kurtsum / datlen}")
        st.write("Hardness를 반올림해 그룹화함")
        st.write("Hardness의 결측치는 drop함")
        st.write("각 그룹의 평균값을 가지고 결측치를 채움")

    with tabs[3]:
        st.subheader("According to Solids")
        df = pd.read_csv('solids_group_improved.csv')
        st.dataframe(data=df.head(10))
        newpar(2)
        skewsum = 0
        kurtsum = 0
        for idx in df.columns:
            if idx == "ph":
                skewsum += skew(2 * df[idx])
                kurtsum += kurtosis(df[idx])
        datlen = len(df.columns)
        st.write(f"Skewness: {skewsum / datlen}")
        st.write(f"Kurtosis: {kurtsum / datlen}")
        st.write("Solids를 반올림해 그룹화함")
        st.write("Solids의 결측치는 drop함")
        st.write("각 그룹의 평균값을 가지고 결측치를 채움")

    # ph_Sulfate_corr 데이터
    with tabs[4]:
        st.subheader("According to ph_Sulfate_corr")
        df = pd.read_csv('ph_Sulfate_corr_clean.csv')
        st.dataframe(data=df.head(10))
        newpar(2)
        skewsum = 0
        kurtsum = 0
        for idx in df.columns:
            if idx == "ph":
                skewsum += skew(2 * df[idx])
                kurtsum += kurtosis(df[idx])
        datlen = len(df.columns)
        st.write(f"Skewness: {skewsum / datlen}")
        st.write(f"Kurtosis: {kurtsum / datlen}")
        st.write("결측값이 있는 Sulfate 데이터에 대해, pH 값이 비슷한 다른 샘플들의 Sulfate 값을 참고하여 결측치를 채우는 방법을 사용함")

    # 5. Solids_Sulfate_corr 데이터
    with tabs[5]:
        st.subheader("According to Solids_Sulfate_corr")
        df = pd.read_csv('Solids_Sulfate_corr.csv')
        st.dataframe(data=df.head(10))
        newpar(2)
        skewsum = 0
        kurtsum = 0
        for idx in df.columns:
            if idx == "ph":
                skewsum += skew(2 * df[idx])
                kurtsum += kurtosis(df[idx])
        datlen = len(df.columns)
        st.write(f"Skewness: {skewsum / datlen}")
        st.write(f"Kurtosis: {kurtsum / datlen}")
        st.write("결측값이 있는 Sulfate 데이터에 대해, Solids 값이 비슷한 다른 샘플들의 Sulfate 값을 참고하여 결측치를 채우는 방법을 사용함")

elif choice=='Model Building':
    st.title(choice)
    data_files = {
        "Fixed Dropna": "fixed_dropna.csv",
        "Hardness Group": "hardness_group_improved.csv",
        "pH Group": "ph_group_improved.csv",
        "pH-Sulfate Correlation": "ph_Sulfate_corr_clean.csv",
        "Solids-Sulfate Correlation": "Solids_Sulfate_corr.csv",
        "Solids Group": "solids_group_improved.csv"
    }


    # 머신러닝 탭 구성
    st.markdown("---")
    
    selected_dataset = st.selectbox("Select Dataset", list(data_files.keys()))
    df = pd.read_csv(data_files[selected_dataset]).dropna()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "Origin" in df.columns:
        mapping = {"river": 0, "lake": 1, "ocean": 2}
        df["Origin"] = df["Origin"].replace(mapping)

    X = df.drop(columns=["Survivability"])
    y = df["Survivability"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    st.title("Machine learning model")
    tabs = st.tabs(
            ["Linear Regression", "Logistic Regression", "Neural Network (MLP)", "Random Forest Classifier"]
        )

    # Linear Regression
    with tabs[0]:
            st.subheader("Linear Regression")
            model = LinearRegression(fit_intercept=True)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = (y_pred > 0.5).astype(int)

            st.write("**Accuracy**:", accuracy_score(y_test, y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

    # Logistic Regression
    with tabs[1]:
            st.subheader("Logistic Regression")
            model = LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000)
            model.fit(X_train_scaled, y_train.astype(int))
            y_pred = model.predict(X_test_scaled)

            st.write("**Accuracy**:", accuracy_score(y_test.astype(int), y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test.astype(int), y_pred))

    # Neural Network (MLP)
    with tabs[2]:
            st.subheader("Neural Network (MLP)")
            model = MLPClassifier(
                hidden_layer_sizes=(100, 100), activation="relu", solver="adam", max_iter=1000, random_state=42
            )
            model.fit(X_train_scaled, y_train.astype(int))
            y_pred = model.predict(X_test_scaled)

            st.write("**Accuracy**:", accuracy_score(y_test.astype(int), y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test.astype(int), y_pred))

    # Random Forest Classifier
    with tabs[3]:
            st.subheader("Random Forest Classifier")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train.astype(int))
            y_pred = model.predict(X_test_scaled)

            st.write("**Accuracy**:", accuracy_score(y_test.astype(int), y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_test.astype(int), y_pred))
elif choice=="Conclusion":
    st.title("Conclusion")
    st.write("과학 이론상 가장 유력했던 pH를 사용하는 방식이 가장 정확도가 높았음")
    st.write("다른 방법을 시도하는 과정에서 유의미한 과정과 논리를 찾는 것에 의미가 있었음")
    st.write("그러나 왜 가장 correlation이 높았던 Solids는 최선의 결과를 내지 못했는지 확인해야 함")
    st.write("Skewness와 kurtosis의 차이가 통계적으로 유의미했는지 고려해 보아야 함(statistical thinking)")
    st.write("Significant Testing이 도움이 될 것으로 예상")