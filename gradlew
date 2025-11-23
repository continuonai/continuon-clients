#!/usr/bin/env sh

DIR="$(cd "$(dirname "$0")" && pwd)"
APP_BASE_NAME=${0##*/}

if [ -z "$JAVA_HOME" ]; then
  JAVA_CMD="java"
else
  JAVA_CMD="$JAVA_HOME/bin/java"
fi

exec "$JAVA_CMD" "-Dorg.gradle.appname=$APP_BASE_NAME" -classpath "$DIR/gradle/wrapper/gradle-wrapper.jar" org.gradle.wrapper.GradleWrapperMain "$@"