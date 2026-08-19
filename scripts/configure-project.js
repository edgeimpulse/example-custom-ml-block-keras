const EdgeImpulseApi = require('edge-impulse-api').EdgeImpulseApi;
const Path = require('node:path');
const fs = require('node:fs');
const program = require('commander');
const inquirer = require('inquirer');

const packageVersion = JSON.parse(fs.readFileSync(Path.join(__dirname, 'package.json'), 'utf-8')).version;
const PROJECT_CONFIG_FILE = Path.join(__dirname, '..', '.ei-project-config.json');
const FOLDER_NAME = Path.basename(Path.join(__dirname, '..'));

program
    .description('Configure an Edge Impulse project and clone a source impulse')
    .version(packageVersion)
    .option('--api-key <apiKey>', 'Edge Impulse project API key')
    .option('--impulse-id <impulseId>', 'Impulse ID to clone')
    .allowUnknownOption(false)
    .parse(process.argv);

// eslint-disable-next-line @typescript-eslint/no-floating-promises
(async () => {
    try {
        console.log('Enter an Edge Impulse API key for a project.');
        console.log('This project should match the type of the custom ML block you want to develop (e.g. image classification, object detection, or plain classification/regression).');
        console.log('');

        const apiKey = (program.apiKey || (await inquirer.prompt([{
            type: 'password',
            name: 'apiKey',
            message: 'API key:',
            mask: '*',
        }])).apiKey).trim();
        if (!apiKey.startsWith('ei_')) {
            throw new Error('API key does not start with "ei_"');
        }

        const api = new EdgeImpulseApi();
        await api.authenticate({
            method: 'apiKey',
            apiKey,
        });

        const apiKeyInfo = await api.projects.getCurrentApiKeyInfo();
        if (!apiKeyInfo.success) {
            throw new Error(apiKeyInfo.error || 'Could not verify API key');
        }
        if (apiKeyInfo.role !== 'admin') {
            throw new Error(`API key must have admin credentials, but has role "${apiKeyInfo.role}"`);
        }

        const projectId = apiKeyInfo.projectId;
        const projects = (await api.projects.listProjects()).projects;
        const project = projects.find(x => x.id === projectId) || projects[0];
        if (project) {
            console.log('Project:', project.owner, '/', project.name);
        }
        else {
            console.log('Project ID:', projectId);
        }

        const impulseRes = await api.impulse.getAllImpulses(projectId);
        if (!impulseRes.success) {
            throw new Error(impulseRes.error || 'Could not list impulses');
        }
        const defaultImpulses = impulseRes.impulses.filter(x => x.type === 'default');
        if (defaultImpulses.length === 0) {
            throw new Error('This project has no default impulses to clone');
        }

        console.log('');
        console.log('Select an impulse to use as the basis for this custom ML block. It will be cloned.');

        const impulse = await selectImpulse(defaultImpulses, program.impulseId);
        const cloneName = `${impulse.name} - clone for ${FOLDER_NAME} - ${((Date.now() - +new Date(2019, 0, 1)).toString(32))}`;

        console.log('');
        console.log(`Cloning impulse "${impulse.name}" as "${cloneName}"...`);
        const cloneJob = await api.impulse.cloneImpulseComplete(projectId, impulse.id, {
            name: cloneName,
        });
        if (!cloneJob.success) {
            throw new Error(cloneJob.error || 'Could not start impulse clone job');
        }

        await api.runJobUntilCompletion({
            type: 'project',
            projectId,
            jobId: cloneJob.id,
        }, data => {
            process.stdout.write(data);
        });

        const clonedImpulseRes = await api.impulse.getAllImpulses(projectId);
        if (!clonedImpulseRes.success) {
            throw new Error(clonedImpulseRes.error || 'Could not list impulses after cloning');
        }

        const clonedImpulse = clonedImpulseRes.impulses.find(x => x.name === cloneName);
        if (!clonedImpulse) {
            throw new Error(`Clone job finished, but could not find cloned impulse named "${cloneName}"`);
        }

        const learnBlock = await selectLearnBlock(clonedImpulse.learnBlocks);

        await fs.promises.writeFile(PROJECT_CONFIG_FILE, JSON.stringify({
            apiKey,
            impulseId: clonedImpulse.id,
            learnBlockId: learnBlock.id,
        }, null, 4) + '\n');

        console.log('');
        console.log(`Cloned impulse ID ${clonedImpulse.id}`);
        console.log(`Selected learn block ID ${learnBlock.id}`);
        console.log(`Wrote ${PROJECT_CONFIG_FILE}`);
    }
    catch (ex) {
        console.error('Failed to configure project:', ex.message || ex);
        process.exitCode = 1;
    }
})();

async function selectImpulse(impulses, selectedImpulseId) {
    if (selectedImpulseId) {
        const impulseId = Number(selectedImpulseId);
        if (!Number.isInteger(impulseId)) {
            throw new Error(`Invalid --impulse-id "${selectedImpulseId}"`);
        }

        const impulse = impulses.find(x => x.id === impulseId);
        if (!impulse) {
            throw new Error(`Could not find impulse with ID ${impulseId}`);
        }
        return impulse;
    }

    if (impulses.length === 1) {
        console.log(`Using the only default impulse: ${impulses[0].name} (ID: ${impulses[0].id})`);
        return impulses[0];
    }

    return (await inquirer.prompt([{
        type: 'list',
        name: 'impulse',
        message: 'Impulse to clone:',
        pageSize: 15,
        choices: impulses.map(impulse => ({
            name: `${impulse.name} (ID: ${impulse.id})`,
            value: impulse,
        })),
    }])).impulse;
}

async function selectLearnBlock(learnBlocks) {
    if (learnBlocks.length === 0) {
        throw new Error('Cloned impulse has no learn blocks');
    }

    if (learnBlocks.length === 1) {
        console.log(`Using the only learn block: ${learnBlocks[0].name} (ID: ${learnBlocks[0].id})`);
        return learnBlocks[0];
    }

    return (await inquirer.prompt([{
        type: 'list',
        name: 'learnBlock',
        message: 'This impulse has multiple learn blocks, select the one that matches the type of ML block you\'re building:',
        pageSize: 15,
        choices: learnBlocks.map(learnBlock => ({
            name: `${learnBlock.name} (ID: ${learnBlock.id})`,
            value: learnBlock,
        })),
    }])).learnBlock;
}
