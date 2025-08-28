@echo off
echo Downloading Gradle Wrapper JAR...
echo.

if not exist "gradle\wrapper" mkdir "gradle\wrapper"

echo Downloading gradle-wrapper.jar for Gradle 9.0 milestone...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/gradle/gradle/raw/v9.0-milestone-1/gradle/wrapper/gradle-wrapper.jar' -OutFile 'gradle\wrapper\gradle-wrapper.jar'"

if exist "gradle\wrapper\gradle-wrapper.jar" (
    echo.
    echo Gradle wrapper downloaded successfully!
    echo You can now open the project in Android Studio.
) else (
    echo.
    echo Failed to download Gradle wrapper.
    echo Please download manually from: https://github.com/gradle/gradle/raw/v9.0-milestone-1/gradle/wrapper/gradle-wrapper.jar
    echo And place it in: gradle\wrapper\gradle-wrapper.jar
)

pause
