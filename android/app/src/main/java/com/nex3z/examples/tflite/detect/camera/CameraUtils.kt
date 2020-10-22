package com.nex3z.examples.tflite.detect.camera

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment


val CAMERA_PERMISSIONS_REQUIRED = arrayOf(Manifest.permission.CAMERA)
const val REQUEST_CODE: Int = 27807


fun Context.hasCameraPermissions(): Boolean = CAMERA_PERMISSIONS_REQUIRED.all {
    ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
}


fun Fragment.hasCameraPermissions(): Boolean = requireContext().hasCameraPermissions()


fun Fragment.requestCameraPermissions(requestCode: Int = REQUEST_CODE) {
    requestPermissions(CAMERA_PERMISSIONS_REQUIRED, requestCode)
}


fun Activity.requestCameraPermissions(requestCode: Int = REQUEST_CODE) {
    ActivityCompat.requestPermissions(this, CAMERA_PERMISSIONS_REQUIRED, requestCode)
}
