<script setup lang="ts">
import { useErrorHandler } from '@/composables/useErrorHandler';
import { loadUserProfile } from '@/composables/useUserProfile';
import { Team, TeamRoleStatus } from '@/services/api';
import { computed, onMounted, ref } from 'vue';

const props = withDefaults(defineProps<{
    modelValue: number | null;
    defaultTeams: Team[];
    defaultSelectedTeam: number | null;
    disabled?: boolean;
    size?: 'normal' | 'large';
}>(), { size: 'normal' });

const emit = defineEmits<{
    (e: 'update:modelValue', value: number | null): void
    (e: 'change', value: number | null): void
}>();

const { handleError } = useErrorHandler();

const model = computed({
    get: () => props.modelValue,
    set: (value) => {
        emit('update:modelValue', value)
        emit('change', value)
    },
})

const options = ref<Team[]>([]);

const loadOptions = async () => {
    try {
        const userProfile = await loadUserProfile();
        options.value = [...props.defaultTeams, ...userProfile!.teams.filter(t => t.status === TeamRoleStatus.ACCEPTED && !props.defaultTeams.some(dt => dt.id === t.id)).map(t => new Team(t.id, t.name))];
    } catch (error) {
        handleError(error);
    }
}

onMounted(async () => {
    await loadOptions();
});

defineExpose({
    refreshTeams: async () => { await loadOptions() }
})

</script>

<template>
    <Select v-model="model" :options="options" option-label="name" option-value="id" :disabled="props.disabled" 
        :class="[{ '!bg-surface-muted !cursor-not-allowed !opacity-60': props.disabled }, props.size === 'large' ? 'h-12 !text-base flex items-center' : '']" class="w-full"
    />
</template>
