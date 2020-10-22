package com.nex3z.examples.tflite.detect.bbox

import android.content.Context
import android.util.AttributeSet
import android.view.LayoutInflater
import android.widget.FrameLayout
import com.nex3z.examples.tflite.detect.R
import kotlinx.android.synthetic.main.view_bbox.view.*


class BBoxView : FrameLayout {

    var label: String?
        set(value) {
            tv_vb_label.text = value
        }
        get() = tv_vb_label.text.toString()

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

    private fun init(attrs: AttributeSet?, defStyle: Int) {
        val inflater = context.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        inflater.inflate(R.layout.view_bbox, this, true)
    }

}