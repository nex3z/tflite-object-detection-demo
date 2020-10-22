package com.nex3z.examples.tflite.detect.bbox

import android.content.Context
import android.util.AttributeSet
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import timber.log.Timber

class BBoxContainerView : FrameLayout {

    constructor(context: Context) : super(context) {
        init(null, 0)
    }

    constructor(context: Context, attrs: AttributeSet) : super(context, attrs) {
        init(attrs, 0)
    }

    constructor(context: Context, attrs: AttributeSet, defStyle: Int)
            : super(context, attrs, defStyle) {
        init(attrs, defStyle)
    }

    private fun init(attrs: AttributeSet?, defStyle: Int) {}

    fun addBoxes(boxes: List<BBox>) {
        var bboxView: BBoxView
        for (box in boxes) {
            bboxView = BBoxView(context)
            bboxView.label = box.label + " %.2f".format(box.confidence)
            val params = LayoutParams((box.width * width).toInt(), (box.height * height).toInt())
            params.leftMargin =  (box.xmin * width).toInt()
            params.topMargin = (box.ymin * height).toInt()
            addView(bboxView, params)
        }
    }

    fun clear() {
        removeAllViews()
    }
}

fun View.setAspectRatio(targetRatio: Float, container: View = this.parent as View) {
    val containerRatio = container.width.toFloat() / container.height
    Timber.v("setTargetRatio(): targetRatio = $targetRatio")
    Timber.v("setTargetRatio(): container = ${container.width} x ${container.height}, container ratio = $containerRatio")

    if (targetRatio > containerRatio) {
        val adjustViewHeight = container.width / targetRatio
        val verticalMargin = ((container.height - adjustViewHeight) / 2).toInt()
        Timber.v("setTargetRatio(): adjustViewHeight = $adjustViewHeight, verticalMargin = $verticalMargin")

        val params = layoutParams as ViewGroup.MarginLayoutParams
        params.topMargin = verticalMargin
        params.bottomMargin = verticalMargin

        requestLayout()
    } else if (targetRatio < containerRatio) {
        val adjustViewWidth = container.height * targetRatio
        val horizontalMargin = ((container.width - adjustViewWidth) / 2).toInt()
        Timber.v("setTargetRatio(): adjustViewWidth = $adjustViewWidth, horizontalMargin = $horizontalMargin")

        val params = layoutParams as ViewGroup.MarginLayoutParams
        params.leftMargin = horizontalMargin
        params.rightMargin = horizontalMargin

        requestLayout()
    }
}