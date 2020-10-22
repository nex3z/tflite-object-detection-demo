package com.nex3z.examples.tflite.detect.bbox

data class BBox(
    val label: String? = null,
    val xmin: Float,
    val ymin: Float,
    val xmax: Float,
    val ymax: Float,
    val confidence: Float = 0.0f
) {

    val width: Float = xmax - xmin
    val height: Float = ymax - ymin

}
