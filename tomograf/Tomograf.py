import math
import numpy as np
import streamlit as st
from PIL import Image


def detector_cords(radius, alpha, fi, n, image_local):
    cords = []
    for i in range(n):
        value = math.radians(alpha + 180 - (fi / 2) + i * (fi / (n - 1)))
        xi = radius * math.cos(value) + image_local.shape[0] // 2
        yi = radius * math.sin(value) + image_local.shape[1] // 2
        cords.append((int(xi), int(yi)))
    return cords


def emitter_cords(radius, alpha, image_local):
    xe = radius * math.cos(math.radians(alpha)) + image_local.shape[0] // 2
    ye = radius * math.sin(math.radians(alpha)) + image_local.shape[1] // 2
    return int(xe), int(ye)


def avg_brightness(line_points, image_local):
    _sum = 0
    width, height = image_local.shape
    counter = 0
    avg = 0
    for x, y in line_points:
        if x < width and y < height:
            _sum += image_local[x][y]
            counter += 1

    if counter != 0:
        avg = _sum // counter

    return avg


def bresenham_line(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    is_steep = abs(dy) > abs(dx)

    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    dx = x2 - x1
    dy = y2 - y1

    error = dx // 2
    y_step = 1 if y1 < y2 else -1

    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx

    if swapped:
        points.reverse()
    return points


@st.cache_data
def radon_transform(number_of_detectors_local, iterations_local, span_local, image_local):
    r = min((image_local.shape[1] // 2), (image_local.shape[0] // 2)) - 1
    sinogram = np.zeros((iterations_local, number_of_detectors_local))
    iterations_sinogram = []
    alpha_steps = np.linspace(0, 359, iterations_local)

    for i, alpha in enumerate(alpha_steps):
        emitter_x, emitter_y = emitter_cords(r, alpha, image_local)
        detectors = detector_cords(r, alpha, span_local, number_of_detectors_local, image_local)
        for j, detector in enumerate(detectors):
            bresenham = bresenham_line(emitter_x, emitter_y, detector[0], detector[1])
            sinogram[i][j] = avg_brightness(bresenham, image_local)

        iterations_sinogram.append(sinogram.copy())
    return sinogram, iterations_sinogram


@st.cache_data
def inverse_radon(sinogram, iterations_local, image_local, span_local, number_of_detectors_local, filters=False):
    if filters:
        for i, sins in enumerate(sinogram):
            sinogram[i] = filtered(sins)

    r = min((image_local.shape[1] // 2), (image_local.shape[0] // 2)) - 1

    width, height = image_local.shape
    output_image = np.zeros((width, height))
    alpha_steps = np.linspace(0, 359, iterations_local)
    image_steps = []

    for i, alpha in enumerate(alpha_steps):
        emitter_x, emitter_y = emitter_cords(r, alpha, image_local)
        detectors = detector_cords(r, alpha, span_local, number_of_detectors_local, image_local)

        for j, detector in enumerate(detectors):
            bresenham = bresenham_line(emitter_x, emitter_y, detector[0], detector[1])
            for x, y in bresenham:
                if 0 <= x < width and 0 <= y < height:
                    output_image[x, y] += sinogram[i, j]
        image_steps.append(output_image.copy())
    return output_image, image_steps


def filtered(sinogram):
    kernel_size = 21
    kernel = []
    for k in range(-kernel_size // 2, kernel_size // 2):
        if k == 0:
            kernel.append(1)
        else:
            if k % 2 != 1:
                kernel.append(0)
            else:
                h = (-4 / (math.pi ** 2)) / (k ** 2)
                kernel.append(h)

    filter_obj = np.convolve(sinogram, kernel, mode='same')
    return filter_obj


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * 255


def root_mean_square_error(original_image, output_image):
    return np.sqrt(np.mean((original_image - output_image) ** 2))


if __name__ == "__main__":
    st.title("Symulator tomografu komputerowego")
    number_of_detectors = st.slider("Liczba detektorów", 60, 360, 120, 10)
    angle = st.slider("Krok alfa emitera", 1, 10, 1, 1)
    span = st.slider("Rozpiętość układu", 10, 270, 120, 10)
    uploaded_file = st.file_uploader("Podaj plik", type=['png', 'jpeg', 'jpg', 'bmp'])
    filter_checkbox = st.checkbox("Włączyć filtrowanie")
    iterations = st.checkbox("Kroki pośrednie")

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False


    def click_button():
        st.session_state.clicked = True


    st.button('Start!', on_click=click_button)

    if st.session_state.clicked and uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        img = np.asarray(image)

        num_of_iterations = int(360 / angle)
        sinus_out, sinogram_parts = radon_transform(number_of_detectors, num_of_iterations, span, img)
        sinus_out = np.array(sinus_out)
        sinus_out_rescaled = normalize(sinus_out)

        end_image, end_image_parts = inverse_radon(sinus_out_rescaled, num_of_iterations, img, span,
                                                   number_of_detectors,
                                                   filter_checkbox)
        end_image = np.array(end_image)
        end_image_rescaled = normalize(end_image)

        st.session_state["end_image"] = end_image_rescaled

        if iterations:
            st.write("RMSE", root_mean_square_error(normalize(img), end_image_rescaled))
            selected_iteration = st.slider("Wybierz iterację", 1, num_of_iterations, num_of_iterations)
            st.image(image, caption="Basic image")

            sin_selected = sinogram_parts[selected_iteration - 1]
            sin_selected = normalize(sin_selected)

            img_selected = end_image_parts[selected_iteration - 1]
            img_selected = normalize(img_selected)

            img_selected = np.uint8(img_selected)
            sin_selected = np.uint8(sin_selected)

            st.image(sin_selected, caption="Sinogram")
            st.image(img_selected, caption="Output image")
        else:
            st.write("RMSE", root_mean_square_error(normalize(img), end_image_rescaled))
            st.image(uploaded_file, caption="Basic image")

            sinus_out_rescaled = np.uint8(sinus_out_rescaled)
            st.image(sinus_out_rescaled, caption="Sinogram")

            end_image_rescaled = np.uint8(end_image_rescaled)
            st.image(end_image_rescaled, caption="Output image")
