module.exports = {
    apps: [
        {
            name: 'cryptoflow-api',
            script: 'api.js',
            cwd: './server',
            instances: 1,
            autorestart: true,
            watch: false,
            max_memory_restart: '500M',
            env: {
                NODE_ENV: 'production',
                PORT: 3001,
                WS_PORT: 3002
            },
            log_date_format: 'YYYY-MM-DD HH:mm:ss',
            error_file: './logs/error.log',
            out_file: './logs/out.log',
            merge_logs: true
        }
    ]
};
