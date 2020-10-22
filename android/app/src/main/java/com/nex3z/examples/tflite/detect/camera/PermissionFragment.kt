package com.nex3z.examples.tflite.detect.camera

import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.nex3z.examples.tflite.detect.R
import timber.log.Timber

class PermissionFragment : Fragment() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (requireContext().hasCameraPermissions()) {
            navigateToCamera()
        } else {
            requestCameraPermissions()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>,
                                            grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Timber.v("onRequestPermissionsResult(): Permission granted")
                navigateToCamera()
            } else {
                Toast.makeText(requireContext(), R.string.m_pf_camera_permission_denied,
                    Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun navigateToCamera() {
        findNavController().navigate(R.id.action_permission_to_camera)
    }
}