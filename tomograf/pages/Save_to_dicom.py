import streamlit as st
from save_dicom import *


if __name__ == "__main__":
    st.title("Zapis do pliku DICOM")
    PatientName = st.text_input("Imię i nazwisko pacjenta: ")
    PatientID = st.text_input("ID pacjenta: ")
    ImageComments = st.text_input("Dodatkowy komentarz: ")
    choose_date = st.date_input("Wybierz datę badania")
    file_name = st.text_input("Podaj nazwę pliku: ")

    data_to_save = {"PatientName": PatientName,
                    "PatientID": PatientID,
                    "ImageComments": ImageComments,
                    "AcquisitionDate": choose_date.strftime("%Y%m%d") if choose_date else ""}

    submit_button = st.button("Submit")

    if "end_image" in st.session_state:
        if submit_button:
            filename = "data/" + file_name + ".dcm"
            save_as_dicom(filename, st.session_state["end_image"], data_to_save)
            st.write("Zapisano zdjęcie")
            del st.session_state["end_image"]
    else:
        st.write("Bardzo nam przykro ale musisz dokonać badania aby można było zapisać plik!")
