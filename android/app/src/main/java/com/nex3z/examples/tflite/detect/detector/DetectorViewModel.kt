package com.nex3z.examples.tflite.detect.detector

import android.annotation.SuppressLint
import android.app.Application
import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.nex3z.examples.tflite.detect.bbox.BBox
import com.nex3z.examples.tflite.detect.util.Timer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import timber.log.Timber
import java.lang.Exception
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.time.TimedValue

class DetectorViewModel(
    application: Application
) : AndroidViewModel(application) {
    val preprocessed: MutableLiveData<Bitmap> = MutableLiveData()
    val result: MutableLiveData<List<BBox>> = MutableLiveData()

    val executor: ExecutorService = Executors.newSingleThreadExecutor()

    private var detector: Detector? = null

    private val converter: RsYuvToRgbConverter = RsYuvToRgbConverter(application)
    private lateinit var imageBuffer: Bitmap

    private lateinit var preprocessor: ImageProcessor

    init {
        viewModelScope.launch {
            try {
                detector = buildClassifier()
            } catch (e: Exception) {
                Timber.e(e, "init(): Failed to build classifier")
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        executor.shutdown()
    }

    private suspend fun buildClassifier(): Detector {
        return withContext(Dispatchers.Default) {
            Timber.v("buildClassifier(): building classifier")
            Detector(
                context = getApplication(),
                model = "model.tflite"
            )
        }
    }

    @SuppressLint("UnsafeExperimentalUsageError")
    fun detect(image: ImageProxy) {
        if (!::imageBuffer.isInitialized) {
            imageBuffer = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
            Timber.v("detect(): size = ${image.width} x ${image.height}, rotationDegrees = ${image.imageInfo.rotationDegrees}")

        }
        if (!::preprocessor.isInitialized) {
            val detector = detector ?: return
            preprocessor = ImageProcessor.Builder()
                .add(ResizeWithCropOrPadOp(detector.inputShape.height, detector.inputShape.width))
                .add(Rot90Op(-image.imageInfo.rotationDegrees / 90))
                .add(NormalizeOp(127.5f, 127.5f))
                .build()
        }

        val timer = Timer()
        converter.yuvToRgb(image.image ?: return, imageBuffer)
        Timber.v("detect(): convert time cost = ${timer.delta()}")
        val processedImage = preprocessor.process(TensorImage.fromBitmap(imageBuffer))
        Timber.v("detect(): preprocess time cost = ${timer.delta()}")
//        preprocessed.postValue(processedImage.bitmap)

        val boxes = detector?.detect(processedImage)
        result.postValue(boxes)
    }
}