pipeline {
    agent any

    environment {
        PYTHON = 'C:\\Users\\admin\\AppData\\Local\\Programs\\Python\\Python313\\python.exe'
    }

    stages {
        stage('Checkout') {
            steps {
                git url: 'https://github.com/AdityaRam24/HMM-.git', branch: 'main'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat """
                   echo Installing dependencies…
                   "%PYTHON%" -m pip install --upgrade pip
                   "%PYTHON%" -m pip install -r requirements.txt
                """
            }
        }

        stage('Run App') {
            steps {
                bat """
                   echo Starting Flask app in background…
                   START /B "" "%PYTHON%" app.py > server.log 2>&1
                   echo Waiting for server to start…
                   TIMEOUT /T 5 /NOBREAK
                   echo Flask should now be running at http://127.0.0.1:5000
                   echo (logs in server.log)
                """
            }
        }
    }

    post {
        always  { echo '🛠 Pipeline finished.' }
        success { echo '✅ Build succeeded!' }
        failure { echo '❌ Build failed!' }
    }
}
