import streamlit as st
from pydicom import dcmread
import datetime


def read_from_dicom(path):
    ds = dcmread(path, force=True)

    st.write("Imię i nazwisko pacjenta: ", ds.PatientName)
    st.write("ID pacjenta: ", ds.PatientID)
    st.write("Komentarz: ", ds.ImageComments)
    try:
        st.write("Data pozyskania zdjęcia: ",
                 datetime.datetime.strptime(ds.AcquisitionDate, "%Y%m%d").strftime("%Y-%m-%d"))
    except:
        st.write("Brak daty pozyskania zdjęcia")
    st.image(ds.pixel_array)


if __name__ == "__main__":
    st.title("Odczyt z pliku")
    dicom_input_file = st.file_uploader("Otwórz plik dicom")

    if dicom_input_file is not None:
        read_from_dicom(dicom_input_file)
