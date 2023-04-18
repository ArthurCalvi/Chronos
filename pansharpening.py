import numpy as np
import rasterio
from rpcm import RPCModel
from skimage.exposure import match_histograms
from skimage.transform import resize
from sklearn.decomposition import PCA

__version__ = "1.0.2dev"


def pansharpen(ms_arr, pan_arr, method='pca', interpolation_order=3, with_nir=True):
    """Pansharpening of images

        Pansharpening is a process of merging high-resolution panchromatic and lower resolution multispectral imagery
        to create a single high-resolution color image. Google Maps and nearly every map creating company use this
        technique to increase image quality. Pansharpening produces a high-resolution color image from three,
        four or more low-resolution multispectral satellite bands plus a corresponding high-resolution panchromatic.
        (Wikipedia)


        :param
        ----------
        ms_arr : str, array of the multi_spectral image
            Raw low resolution multispectral image,
            Of size (C, H1, W1) C = 3 or 4 depending on the presence of the NIR (Near-InfraRed) band

        pan_arr : str, array of pan image
            High Resolution panchromatic image, size (1, H2, W2)

        method : str, {'pca', 'ihs'} default 'pca'
            Pansharpening method

        interpolation_order : int, optional by default = 3,  {1, 2, 3, 4, 5}
                    0: Nearest-neighbor
                    1: Bi-linear
                    2: Bi-quadratic
                    3: Bi-cubic (default)
                    4: Bi-quartic
                    5: Bi-quintic

        with_nir : bool, default True
            Include the Near-InfraRed band in the pansharpening process

        References
        ----------
        For IHS :  See Image Fusion: Theories, Techniques and Applications,
            H.B. Mitchell (https://bit.ly/2ZdplLo) for details
             about the transform

        For PCA : See Review of the Pansharpening Methods for Remote Sensing Images
            Meng et al. (https://bit.ly/325yHMA) for details
        --------
        """

    # Resizing the images + Rescaling

    raw = np.transpose(ms_arr, (1, 2, 0))  # (4, h2, w2) --> (h2, w2, 4)
    pan_s = pan_arr.squeeze()  # (1, h1, w1) --> (h1, w1)

    rgbi_resized = resize(raw, pan_s.shape, order=interpolation_order, preserve_range=True)  # RGBI
    rgb_resized = rgbi_resized[:, :, :3]

    r = rgb_resized[:, :, 0]
    g = rgb_resized[:, :, 1]
    b = rgb_resized[:, :, 2]
    nir = rgbi_resized[:, :, -1]

    h1, w1 = pan_s.shape
    channels = 4 if with_nir else 3
    color_resized = rgbi_resized if with_nir else rgb_resized
    # Pansharpening
    if method == 'pca':
        """ PCA method was tested on RGBI image, but feel free to try it on RGB image alone
        Here I'm keeping the choice to you 
        """
        pca = PCA(n_components=channels)

        color_resized = np.reshape(color_resized, (h1 * w1, channels))

        mask_band = ~np.all(color_resized == 0, axis=-1)
        color_no_band = color_resized[mask_band]

        pca_hs = pca.fit_transform(color_no_band)
        pan_factor = np.sign(pca.components_[0][0])

        # # Taking the first principal component :
        intensity = pca_hs[:, 0]

        # # Switching between intensity and the panchromatic band

        pan_s = np.reshape(pan_s, (h1 * w1, 1))
        pan_new = pan_factor * (pan_s[mask_band] - np.mean(pan_s[mask_band])) * np.std(intensity, ddof=1) / np.std(
            pan_s[mask_band], ddof=1) + np.mean(intensity)

        pca_hs[:, 0] = pan_new[:, 0]

        pansharpened = np.zeros((h1 * w1, channels))
        pansharpened[mask_band] = pca.inverse_transform(pca_hs)
        pansharpened = np.reshape(pansharpened, (h1, w1, channels))

        # Re-centering to match the raw
        pansharpened = pansharpened - np.mean(pansharpened, axis=(0, 1)) + np.mean(color_resized)

    elif method == 'ihs':

        """ IHS method was tested to return RGB image """

        # Projection to ihs space
        # This is equivalent to the IHS transform that can be found on certain packages

        root = np.sqrt(2)
        mat1 = np.array([[1 / 3, 1 / 3, 1 / 3], [-root / 6, -root / 6, 2 * root / 6], [1 / root, -1 / root, 0]])
        r_flat, g_flat, b_flat = r.flatten(), g.flatten(), b.flatten()

        intensity_flat, v1, v2 = np.dot(mat1, [r_flat, g_flat, b_flat])

        intensity = np.reshape(intensity_flat, r.shape)
        # Histogram matching

        pan_matched = match_histograms(pan_s, intensity, multichannel=True)

        # Re-projecting back to the main space

        mat2 = np.array([[1, -1 / root, -1 / root], [1, -1 / root, -1 / root], [1, root, 0]])
        r_new_flat, g_new_flat, b_new_flat = np.dot(mat2, [pan_matched.flatten(), v1, v2])

        r_new = np.reshape(r_new_flat, r.shape)
        g_new = np.reshape(g_new_flat, r.shape)
        b_new = np.reshape(b_new_flat, r.shape)

        if with_nir:
            pansharpened = np.array([r_new, g_new, b_new, nir]).transpose(1, 2, 0)
        else:
            pansharpened = np.array([r_new, g_new, b_new]).transpose(1, 2, 0)

        pansharpened = pansharpened - np.mean(pansharpened, axis=(0, 1)) + np.mean(color_resized)

    else:
        raise ValueError("Invalid pansharpening method {}".format(method))

    # Rescaling
    mask = rgbi_resized[:, :, :(3 + with_nir)] != 0
    pansharpened_new = np.where(mask, pansharpened, 0)

    return pansharpened_new.transpose(2, 0, 1)


def histogram_match_pansharpen_visual(pan_image, visual):

    #nan ? 

    # Read & Mask visual
    pan_image = pan_image.transpose(1, 2, 0)
    out_visual = visual.transpose(1, 2, 0)[:, :, :pan_image.shape[2]]  

    return match_histograms(pan_image, out_visual, channel_axis=2).transpose(2, 0, 1)

