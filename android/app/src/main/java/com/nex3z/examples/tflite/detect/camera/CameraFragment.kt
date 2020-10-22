package com.nex3z.examples.tflite.detect.camera

import android.os.Bundle
import android.util.DisplayMetrics
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.nex3z.examples.tflite.detect.R
import com.nex3z.examples.tflite.detect.bbox.setAspectRatio
import com.nex3z.examples.tflite.detect.detector.DetectorViewModel
import kotlinx.android.synthetic.main.camera_fragment.*
import timber.log.Timber
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class CameraFragment : Fragment() {
    private lateinit var viewModel: DetectorViewModel

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?,
                              savedInstanceState: Bundle?): View? {
        return inflater.inflate(R.layout.camera_fragment, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        viewModel = ViewModelProvider(this).get(DetectorViewModel::class.java)
        init()
    }

    private fun init() {
        initView()
        bindCamera()
        bindData()
    }

    private fun initView() {
        bbcv_cf_bbox_container.post {
            bbcv_cf_bbox_container.setAspectRatio(1.0f)
        }
    }

    private fun bindData() {
        viewModel.preprocessed.observe(viewLifecycleOwner) {
            iv_cf_preview.setImageBitmap(it)
        }

        viewModel.result.observe(viewLifecycleOwner) {
            val boxes = it.filter { box -> box.confidence >= 0.8f }
            if (boxes.isNotEmpty()) {
                bbcv_cf_bbox_container.clear()
                bbcv_cf_bbox_container.addBoxes(boxes)
            }
        }
    }

    private fun bindCamera() = pv_cf_view_finder.post {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener({
            val metrics = DisplayMetrics().also { pv_cf_view_finder.display.getRealMetrics(it) }
            val ratio = aspectRatio(metrics.widthPixels, metrics.heightPixels)
            val rotation = pv_cf_view_finder.display.rotation
            Timber.d("bindCamera(): metrics = $metrics, ratio = $ratio, rotation = $rotation")

            val cameraProvider = cameraProviderFuture.get()

            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build()

            val preview = Preview.Builder()
                .setTargetAspectRatio(ratio)
                .setTargetRotation(rotation)
                .build()

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(viewModel.executor, { image ->
                        viewModel.detect(image)
                        image.close()
                    })
                }

            cameraProvider.unbindAll()

            try {
                val camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
                Timber.i("bindCamera(): sensorRotationDegrees = ${camera.cameraInfo.sensorRotationDegrees}")
                preview.setSurfaceProvider(pv_cf_view_finder.surfaceProvider)
            } catch (e: Exception) {
                Timber.e(e, "bindCamera(): Failed to bind use cases")
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    companion object {
        private const val RATIO_4_3_VALUE = 4.0 / 3.0
        private const val RATIO_16_9_VALUE = 16.0 / 9.0

        fun aspectRatio(width: Int, height: Int): Int {
            val previewRatio = max(width, height).toDouble() / min(width, height)
            if (abs(previewRatio - RATIO_4_3_VALUE) <= abs(previewRatio - RATIO_16_9_VALUE)) {
                return AspectRatio.RATIO_4_3
            }
            return AspectRatio.RATIO_16_9
        }
    }
}