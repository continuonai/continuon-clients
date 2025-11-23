@ECHO OFF
SET DIR=%~dp0
SET APP_BASE_NAME=%~n0
SET JAVA_EXE=java
IF DEFINED JAVA_HOME SET JAVA_EXE=%JAVA_HOME%\bin\java.exe
"%JAVA_EXE%" -Dorg.gradle.appname=%APP_BASE_NAME% -classpath "%DIR%\gradle\wrapper\gradle-wrapper.jar" org.gradle.wrapper.GradleWrapperMain %*