module.exports = {
    apps: [
        {
            name: 'api-server',
            script: 'api.js',
            cwd: 'C:\\CryptoFlow\\server',
            instances: 1,
            autorestart: true,
            watch: true,
            ignore_watch: ['node_modules', 'logs', 'db.sqlite', 'db.sqlite-journal'],
            env: {
                NODE_ENV: 'production',
                PORT: 7071
            },
            log_date_format: 'YYYY-MM-DD HH:mm:ss'
        },
        {
            name: 'caddy-proxy',
            script: 'C:\\Caddy\\caddy.exe',
            args: 'run --config C:\\Caddy\\Caddyfile',
            binary: 'C:\\Caddy\\caddy.exe',
            cwd: 'C:\\Caddy',
            instances: 1,
            autorestart: true,
            watch: false
        }
    ]
};
