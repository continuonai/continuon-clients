#
# Generated file, do not edit.
#

Pod::Spec.new do |s|
  s.name             = 'FlutterPluginRegistrant'
  s.version          = '0.0.1'
  s.summary          = 'Registers plugins with your Flutter app'
  s.description      = <<-DESC
Depends on all your plugins, and provides a function to register them.
                       DESC
  s.homepage         = 'https://flutter.dev'
  s.license          = { :type => 'BSD' }
  s.author           = { 'Flutter Dev Team' => 'flutter-dev@googlegroups.com' }
  s.ios.deployment_target = '12.0'
  s.source_files =  "Classes", "Classes/**/*.{h,m}"
  s.source           = { :path => '.' }
  s.public_header_files = './Classes/**/*.h'
  s.static_framework    = true
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  s.dependency 'Flutter'
  s.dependency 'background_downloader'
  s.dependency 'cloud_firestore'
  s.dependency 'firebase_auth'
  s.dependency 'firebase_core'
  s.dependency 'flutter_blue_plus_darwin'
  s.dependency 'flutter_gemma'
  s.dependency 'flutter_secure_storage'
  s.dependency 'google_sign_in_ios'
  s.dependency 'integration_test'
  s.dependency 'large_file_handler'
  s.dependency 'mobile_scanner'
  s.dependency 'nsd_ios'
  s.dependency 'path_provider_foundation'
  s.dependency 'permission_handler_apple'
  s.dependency 'shared_preferences_foundation'
  s.dependency 'url_launcher_ios'
end
