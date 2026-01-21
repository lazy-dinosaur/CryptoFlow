/**
 * PM2 Ecosystem Configuration
 *
 * Usage:
 *   pm2 start ecosystem.config.cjs              # Start all services
 *   pm2 start ecosystem.config.cjs --only api   # Start API only
 *   pm2 restart api                             # Restart API
 *   pm2 logs                                    # View all logs
 *   pm2 status                                  # Check status
 */

module.exports = {
    apps: [
        {
            // Candle Collector - fetches and stores candles from exchanges
            name: 'collector',
            script: 'collector-standalone.js',
            cwd: './server',
            instances: 1,
            autorestart: true,
            watch: false,
            max_memory_restart: '300M',
            env: {
                NODE_ENV: 'production'
            },
            log_date_format: 'YYYY-MM-DD HH:mm:ss',
            error_file: './logs/collector-error.log',
            out_file: './logs/collector-out.log',
            merge_logs: true
        },
        {
            // API Server
            name: 'api',
            script: 'api.js',
            cwd: './server',
            instances: 1,
            autorestart: true,
            watch: false,
            max_memory_restart: '300M',
            env: {
                NODE_ENV: 'production',
                PORT: 3000
            },
            log_date_format: 'YYYY-MM-DD HH:mm:ss',
            error_file: './logs/api-error.log',
            out_file: './logs/api-out.log',
            merge_logs: true
        },
        {
            // Paper Trading - channel-based strategy
            name: 'paper-trading',
            script: 'ml_paper_trading.py',
            cwd: './server',
            interpreter: 'python3',
            interpreter_args: '-u',
            instances: 1,
            autorestart: true,
            watch: false,
            max_memory_restart: '500M',
            env: {
                NODE_ENV: 'production',
                PYTHONUNBUFFERED: '1'
            },
            log_date_format: 'YYYY-MM-DD HH:mm:ss',
            error_file: './logs/paper-trading-error.log',
            out_file: './logs/paper-trading-out.log',
            merge_logs: true
        }
    ]
};
