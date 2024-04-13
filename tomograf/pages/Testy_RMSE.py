from Tomograf import radon_transform, inverse_radon, normalize, root_mean_square_error
import streamlit as st
import numpy as np
from PIL import Image


def RMSE_test(image, checking_now, current_step, number_of_det=180, iterations=180, span=180, filters=False):
    sinus_out, sinogram_parts = radon_transform(number_of_det, iterations, span, image)
    sinus_out = np.array(sinus_out)
    sinus_out_rescaled = normalize(sinus_out)

    end_image, end_image_parts = inverse_radon(sinus_out_rescaled, iterations, image, span, number_of_det,
                                               filters)
    end_image = np.array(end_image)
    end_image_rescaled = normalize(end_image)
    rmse = root_mean_square_error(normalize(image), end_image_rescaled)

    end_image_rescaled = np.uint8(end_image_rescaled)
    return end_image_rescaled, f'{checking_now}: {current_step}\nRMSE: {rmse:.6f}', rmse


if __name__ == "__main__":
    st.title("Testowanie błędu średniokwardratowego")
    uploaded_file = st.file_uploader("Podaj plik", type=['png', 'jpeg', 'jpg', 'bmp'])

    submit = st.button('Start!')
    if submit and uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("L")
        img = np.asarray(original_image)

        st.write("Zmienna liczba detektorów")
        cols = st.columns(4)
        detectors_values = np.arange(90, 720 + 1, 90)
        for idx, x in enumerate(detectors_values):
            end_img, text_under, rmse_local = RMSE_test(img, "Detektory", x, number_of_det=x)
            cols[idx % 4].image(end_img, width=160)
            cols[idx % 4].text(text_under)
            st.cache_data.clear()

        st.write("Zmienna liczba iteracji")
        cols2 = st.columns(4)
        scan_values = np.arange(90, 720 + 1, 90)
        for idx, y in enumerate(scan_values):
            end_img, text_under, rmse_local = RMSE_test(img, "Iteracja", y, iterations=y)
            cols2[idx % 4].image(end_img, width=160)
            cols2[idx % 4].text(text_under)
            st.cache_data.clear()

        st.write("Zmienna liczba rozpiętości wachlarza")
        cols3 = st.columns(3)
        scan_values = np.arange(45, 270 + 1, 45)
        for idx, z in enumerate(scan_values):
            end_img, text_under, rmse_local = RMSE_test(img, "Rozpiętość", z, span=z)
            cols3[idx % 3].image(end_img, width=160)
            cols3[idx % 3].text(text_under)
            st.cache_data.clear()
