import os
import time
import multiprocessing
import numpy as np
from scipy import ndimage
from scipy import optimize
import scipy.sparse.csgraph
import pyfftw
import warnings
warnings.simplefilter("ignore", optimize.OptimizeWarning)
from collections import Counter

import rasterio
warnings.filterwarnings("ignore",
                        category=rasterio.errors.NotGeoreferencedWarning)
from itsr import parallel, utils


# ITSR_POOL_TYPE must have one of the following values: threads or disable
POOL_TYPE = os.environ.get('ITSR_POOL_TYPE', 'threads')
# NB_READERS: number of parallel workers used to read the input files
NB_READERS = os.environ.get('ITSR_NB_READERS', multiprocessing.cpu_count())
# NB_WORKERS: number of parallel workers used to process images
NB_WORKERS = os.environ.get('ITSR_NB_WORKERS', multiprocessing.cpu_count())

from itsr.registration import compute_registering_shifts, compute_registering_shifts_with_references, apply_shift, read_images

def main(inputs, outputs, references=None, max_shift=None,
         second_max_threshold=0.6, pc_threshold=0, verbose=True, timer=False,
         single_shift=False, timeout=60, use_leprince=False,
         extra_inputs=None, extra_outputs=None):
    """
    Args:
        inputs: list of paths (or list of lists of paths) to the bands of the
            input images
        outputs: list of paths (or list of lists of paths) to the bands of the
            output images
        references: optional, list of paths (or list of lists of paths) to the
            bands of the reference (already registered) images
        max_shift: maximum shift allowed for any pairwise estimated shift.
            Larger shifts are discarded.
        second_max_threshold
        pc_threshold
        verbose: print non-critical information (progress for example)
        timer: print time spent reading and registering images
        single_shift (bool): if True, the same averaged shift is used to
            register all the input images.
        timeout (int): number of seconds allowed per function call when parallelizing
        use_leprince (bool): if True, the phase correlation is improved by
            masking some frequencies
        extra_inputs: list of paths (or list of lists of paths), optional
            path to extra bands for the input images. These are not used to
            compute the registering shifts, but the shifts are applied to them.
        extra_outputs: list of paths (or list of lists of paths), optional
            path to the bands of the registered extra inputs

    The input images may have multiple channels in the same file.
    """
    # read images
    t1 = time.time()
    # arrays is a list of lists of 2D numpy masked arrays
    # profiles and tags are also list of lists
    arrays, profiles, tags = read_images(inputs + references if references else inputs,
                                         verbose=verbose, timeout=timeout)
    input_arrays = arrays[:len(inputs)]
    profiles = profiles[:len(inputs)]
    tags = tags[:len(inputs)]
    reference_arrays = arrays[len(inputs):]  # empty array if no references

    # re-organise (extra) inputs and outputs as list of lists
    if isinstance(inputs[0], str):
        inputs = [[p] for p in inputs]
        outputs = [[p] for p in outputs]

    # read nodata values (and set them if not defined)
    for pp, ii in zip(profiles, input_arrays):
        for p, i in zip(pp, ii):
            if p['nodata'] is None:
                # np.inexact include floating and complexfloating types
                p['nodata'] = np.nan if np.issubdtype(i.dtype, np.inexact) else 0
    nodata_vals = [p[0]['nodata'] for p in profiles]

    t2 = time.time()

    # compute the module of complex arrays
    input_arrays_abs = [[np.abs(b) if np.iscomplexobj(b) else b for b in a]
                        for a in input_arrays]

    # run the registration algorithm
    if references is None:
        shifts = compute_registering_shifts(
                input_arrays_abs,
                max_shift=max_shift,
                second_max_threshold=second_max_threshold,
                pc_threshold=pc_threshold,
                verbose=verbose,
                timeout=timeout,
                use_leprince=use_leprince)
    else:
        # compute the module of complex arrays
        reference_arrays = [[np.abs(b) if np.iscomplexobj(b) else b for b in a]
                            for a in reference_arrays]
        shifts = compute_registering_shifts_with_references(
                input_arrays_abs,
                reference_arrays,
                single_shift=single_shift,
                max_shift=max_shift,
                second_max_threshold=second_max_threshold,
                pc_threshold=pc_threshold,
                verbose=verbose,
                timeout=timeout,
                use_leprince=use_leprince)

    t3 = time.time()

    if nodata_vals is None:
        # this will later be converted to zeros if inputs_arrays are not float
        nodata_vals = [np.nan for i in input_arrays]

    # apply shifts
    # output_arrays is a list of lists of numpy masked arrays
    if verbose:
        print('resampling images... ')

    output_arrays = parallel.run_calls(apply_shift,
                                       list(zip(input_arrays,
                                                shifts,
                                                nodata_vals)),
                                       verbose=verbose,
                                       pool_type=POOL_TYPE,
                                       nb_workers=NB_WORKERS,
                                       timeout=timeout)

    t4 = time.time()

    # get extra metadata
    version_number = utils.get_version_number()
    git_revision_hash = utils.get_git_revision_hash()

    # write the registered images and handle the geotiff metadata
    for bands_profiles, bands_tags, bands_arrays, shift, output_paths, input_paths \
            in zip(profiles, tags, output_arrays, shifts, outputs, inputs):

        # skip images that couldn't be registered
        if np.isnan(shift).any():
            if len(input_paths) == 1:
                print('WARNING: image {} not registered'.format(input_paths[0]))
            else:
                print('WARNING: image {} and its {} associated bands not '
                      'registered'.format(input_paths[0], len(input_paths)-1))
            continue

        # add extra metadata
        extra_metadata = {'REGISTRATION_SHIFT': ' '.join(str(t) for t in shift)}
        if version_number is not None:
            extra_metadata.update({'ITSR_VERSION': version_number})
        if git_revision_hash is not None:
            extra_metadata.update({'ITSR_GIT_REVISION': git_revision_hash})

        # multiband image
        if len(output_paths) == 1 and len(bands_arrays) > 1:
            bands_arrays = [np.ma.array(bands_arrays).squeeze()]

        # write registered images
        for band_profile, band_tags, band_array, path in zip(
                bands_profiles, bands_tags, bands_arrays, output_paths):
            band_tags.update(extra_metadata)
            path = os.path.abspath(os.path.expanduser(path))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            utils.rio_write(path, band_array, band_profile, band_tags)

    if timer:
        print("Reading images: %.3fms" % ((t2-t1)*1000))
        print("Computing registering shifts: %.3fms" % ((t3-t2)*1000))
        print("Resampling images: %.3fms" % ((t4-t3)*1000))


    ######################## MODIF HERE ########################
    #extra inputs
    if extra_inputs:
        for i in range(len(extra_inputs)):

            
            if isinstance(extra_inputs[i][0], str):
                extra_input = [[p] for p in extra_inputs[i]]
                extra_output = [[p] for p in extra_outputs[i]]
            else :
                extra_output = extra_outputs[i]
                extra_input = extra_inputs[i]

            # read extra bands (into a list of lists of 2D numpy masked arrays)
            extra_input_arrays, extra_profiles, extra_tags = read_images(
                    extra_input, verbose=verbose, timeout=timeout)
            # extra bands should be available for all images
            assert len(input_arrays) == len(extra_input_arrays)
            # add extra input bands to input bands
            input_arrays = extra_input_arrays
            profiles = extra_profiles
            tags = extra_tags
            inputs = extra_input
            outputs = extra_output 

            # print('inputs (extra): ', inputs)
            # print('outputs (extra): ', outputs)

            if nodata_vals is None:
                # this will later be converted to zeros if inputs_arrays are not float
                nodata_vals = [np.nan for i in input_arrays]

            # apply shifts
            # output_arrays is a list of lists of numpy masked arrays
            if verbose:
                print('resampling images... ')
            output_arrays = parallel.run_calls(apply_shift,
                                            list(zip(input_arrays,
                                                        shifts,
                                                        nodata_vals)),
                                            verbose=verbose,
                                            pool_type=POOL_TYPE,
                                            nb_workers=NB_WORKERS,
                                            timeout=timeout)

            t4 = time.time()

            # get extra metadata
            version_number = utils.get_version_number()
            git_revision_hash = utils.get_git_revision_hash()

            # write the registered images and handle the geotiff metadata
            for bands_profiles, bands_tags, bands_arrays, shift, output_paths, input_paths \
                    in zip(profiles, tags, output_arrays, shifts, outputs, inputs):

                # skip images that couldn't be registered
                if np.isnan(shift).any():
                    if len(input_paths) == 1:
                        print('WARNING: image {} not registered'.format(input_paths[0]))
                    else:
                        print('WARNING: image {} and its {} associated bands not '
                            'registered'.format(input_paths[0], len(input_paths)-1))
                    continue

                # add extra metadata
                extra_metadata = {'REGISTRATION_SHIFT': ' '.join(str(t) for t in shift)}
                if version_number is not None:
                    extra_metadata.update({'ITSR_VERSION': version_number})
                if git_revision_hash is not None:
                    extra_metadata.update({'ITSR_GIT_REVISION': git_revision_hash})

                # multiband image
                if len(output_paths) == 1 and len(bands_arrays) > 1:
                    bands_arrays = [np.ma.array(bands_arrays).squeeze()]

                # write registered images
                for band_profile, band_tags, band_array, path in zip(
                        bands_profiles, bands_tags, bands_arrays, output_paths):
                    band_tags.update(extra_metadata)
                    path = os.path.abspath(os.path.expanduser(path))
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    utils.rio_write(path, band_array, band_profile, band_tags)

            if timer:
                print("Reading images: %.3fms" % ((t2-t1)*1000))
                print("Computing registering shifts: %.3fms" % ((t3-t2)*1000))
                print("Resampling images: %.3fms" % ((t4-t3)*1000))
