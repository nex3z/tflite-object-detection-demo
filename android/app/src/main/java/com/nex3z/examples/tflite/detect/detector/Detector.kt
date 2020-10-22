package com.nex3z.examples.tflite.detect.detector

import android.content.Context
import android.util.Size
import com.nex3z.examples.tflite.detect.bbox.BBox
import com.nex3z.examples.tflite.detect.util.Timer
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.TensorImage
import timber.log.Timber
import java.io.Closeable


class Detector(
    context: Context,
    model: String = "model.tflite",
    device: Device = Device.GPU,
    numThreads: Int = 4
) {

    private val delegate: Delegate? = when(device) {
        Device.CPU -> null
        Device.NNAPI -> NnApiDelegate()
        Device.GPU -> GpuDelegate()
    }

    private val interpreter: Interpreter = Interpreter(
        FileUtil.loadMappedFile(context, model),
        Interpreter.Options().apply {
            setNumThreads(numThreads)
            delegate?.let { addDelegate(it) }
        }
    )

    val inputShape: Size = with(interpreter.getInputTensor(0)) {
        val shape = shape()
        Size(shape[2], shape[1])
    }

    private val inputBuffer: TensorImage =
        with(interpreter.getInputTensor(0)) {
            Timber.v("[input] shape = ${shape()?.contentToString()} dataType = ${dataType()}")
            TensorImage(dataType())
        }

    private val outputLocationsBuffer: Array<Array<FloatArray>> = with(interpreter.getOutputTensor(0)) {
        val shape = shape()
        Timber.v("[output] boxes shape = ${shape.contentToString()}, dataType = ${dataType()}")
        Array(1) { Array(shape[1]) { FloatArray(shape[2]) } }
    }

    private val outputClassesBuffer: Array<FloatArray> =  with(interpreter.getOutputTensor(1)) {
        val shape = shape()
        Timber.v("[output] classes shape = ${shape.contentToString()}, dataType = ${dataType()}")
        Array(1) { FloatArray(shape[1]) }
    }

    private val outputScoresBuffer: Array<FloatArray> =  with(interpreter.getOutputTensor(2)) {
        val shape = shape()
        Timber.v("[output] scores shape = ${shape.contentToString()}, dataType = ${dataType()}")
        Array(1) { FloatArray(shape[1]) }
    }

    private val numDetectionsBuffer: FloatArray = with(interpreter.getOutputTensor(3)) {
        val shape = shape()
        Timber.v("[output] num detections shape = ${shape.contentToString()}, dataType = ${dataType()}")
        FloatArray(shape[0])
    }

    private val outputMap: Map<Int, Any> = hashMapOf(
        0 to outputLocationsBuffer,
        1 to outputClassesBuffer,
        2 to outputScoresBuffer,
        3 to numDetectionsBuffer
    )

    fun detect(image: TensorImage): List<BBox> {
        val timer = Timer()
        interpreter.runForMultipleInputsOutputs(arrayOf(image.buffer), outputMap)
        Timber.v("detect(): time cost = ${timer.stop()}")
        Timber.v("detect(): locations = ${outputLocationsBuffer.contentDeepToString()}")
        Timber.v("detect(): classes = ${outputClassesBuffer.contentDeepToString()}")
        Timber.v("detect(): scores = ${outputScoresBuffer.contentDeepToString()}")
        Timber.v("detect(): numDetections = ${numDetectionsBuffer.contentToString()}")

        return (0 until numDetectionsBuffer[0].toInt()).map {
            BBox(
                label = outputClassesBuffer[0][it].toString(),
                ymin = outputLocationsBuffer[0][it][0],
                xmin = outputLocationsBuffer[0][it][1],
                ymax = outputLocationsBuffer[0][it][2],
                xmax = outputLocationsBuffer[0][it][3],
                confidence = outputScoresBuffer[0][it]
            )
        }.toList()
    }

    fun close() {
        interpreter.close()
        if (delegate is Closeable) {
            delegate.close()
        }
    }
}